import argparse
import json
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time
import shortuuid
from tqdm import tqdm
from PIL import Image

#try:
from eagle_eye.model.utils import *
from eagle_eye.model.ee_model import EeModel
from eagle_eye.model.kv_cache import initialize_past_key_values
from eagle_eye.model.choices import *


def ee_forward(input_ids, pixel_values, image_grid_thw, model, tokenizer, tree_choices, logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.ee_layer.reset_kv()

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices,
            device=model.language_model.layers[-1].self_attn.q_proj.weight.device,
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.lm_head.weight.device
        )
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.language_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_tree_mode(model.language_model)

    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids=input_ids,
        pixel_values=pixel_values,
        model=model,
        tree_attn_mask=tree_buffers["tree_attn_mask"],
        past_key_values=past_key_values,
        logits_processor=logits_processor,
        image_grid_thw=image_grid_thw,
    )
    new_token = 0

    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits=tree_logits,
            tree_indices=tree_buffers["tree_indices"],
            retrieve_indices=tree_buffers["retrieve_indices"],
            sample_token=sample_token,
            logits_processor=logits_processor,
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model=model,
            tree_candidates=tree_candidates,
            past_key_values=past_key_values,
            tree_position_ids=tree_buffers["tree_position_ids"],
            input_ids=input_ids,
            retrieve_indices=tree_buffers["retrieve_indices_head"],
        )
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits=logits,
            candidates=candidates,
            logits_processor=logits_processor,
            cart_candidates_prob=cart_candidates_prob,
            op=tree_logits[2],
            p_indices=tree_buffers["p_indices"],
            tree_candidates=tree_candidates,
            b_indices=tree_buffers["b_indices"],
        )
        input_ids,tree_logits,new_token,hidden_state,sample_token = update_inference_inputs(
            input_ids=input_ids,
            candidates=candidates,
            best_candidate=best_candidate,
            accept_length=accept_length,
            retrieve_indices=tree_buffers["retrieve_indices"],
            logits_processor=logits_processor,
            logits=logits,
            tree_logits=tree_logits,
            new_token=new_token,
            past_key_values_data_list=past_key_values_data,
            current_length_data=current_length_data,
            model=model,
            hidden_state=hidden_state,
            hidden_state_new=hidden_state_new,
            sample_p=sample_p,
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        if input_ids.shape[1] > 4096:
            break
    return input_ids, new_token, idx


def run_eval(
        base_model_path,
        ee_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        datapath,
        max_new_token,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    
    with open(os.path.expanduser(question_file), "r") as f:
        js = json.load(f)

    images = js.get("images", [])
    
    data=images[question_begin:question_end]
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(data) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(data), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ee_model_path,
                model_id,
                data[i: i + chunk_size],
                answer_file,
                datapath,
                max_new_token,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                tree_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ee_model_path,
        model_id,
        questions,
        answer_file,
        datapath,
        max_new_token,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    #temperature = 0.0

    model = EeModel.from_pretrained(
        base_model_path=base_model_path,
        ee_model_path=ee_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager"
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('warmup ...')
    # warmup
    for j in range(3):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Provide a detailed description of the given image."},
                ],
            }
        ]
                
        text = model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 
        
        
        image = Image.open(os.path.join(datapath, questions[j]['file_name']))

        inputs = model.processor(images=image, text=text, return_tensors="pt")
        # qs = question["turns"][j]
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt() + " "
        input_ids = inputs.input_ids
        pixel_values=inputs.pixel_values
        image_grid_thw=inputs.image_grid_thw
        # try:
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = ee_forward(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            image_grid_thw=torch.as_tensor(image_grid_thw).cuda(),
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
    print('Warmup done')

    torch.manual_seed(123)
    for question in tqdm(questions):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": "Provide a detailed description of the given image."},
                ],
            }
        ]
                
        text = model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 
        
        
        image = Image.open(os.path.join(datapath, questions[j]['file_name']))

        inputs = model.processor(images=image, text=text, return_tensors="pt")
        
        input_ids = inputs.input_ids
        pixel_values=inputs.pixel_values
        image_grid_thw = inputs.image_grid_thw
        
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, new_token, idx = ee_forward(
            input_ids=torch.as_tensor(input_ids).cuda(),
            pixel_values=torch.as_tensor(pixel_values).cuda(),
            image_grid_thw=torch.as_tensor(image_grid_thw).cuda(),
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        output_ids = output_ids[0][len(input_ids[0]):]
        new_token=output_ids.shape[-1]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        if tokenizer.eos_token and output.find(tokenizer.eos_token) > 0:
            output = output[: output.find(tokenizer.eos_token)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        
        # Dump answers
    
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "response":output,
                "idx": idx,
                "new_tokens": new_token,
                "wall_time": total_time,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ee-model-path",
        type=str,
        default="/home/wzy/eagle/EAGLE_EYE/eagle_eye/weights/qwen2.5vl_video/state_0",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/wzy/eagle/EAGLE_EYE/Qwen2.5-VL-7B-Instruct",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="Qwen2.5-VL-7B-Instruct-fp16-ee")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="COCO-caption",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        default=0,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", 
        type=int, 
        default=100,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-dir", type=str,default="../outputs",help="The output answer file.")
    parser.add_argument("--datajson", type=str, default="./captions_val2014.json",help="Path to the input JSON file containing questions or data.")  
    parser.add_argument("--datapath", type=str, default="/home/share/coco_data/val2014",help="Name or path of the dataset to be used for evaluation.")  

    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    question_file=args.datajson
    datapath=args.datapath
    

    args.tree_choices = eval(args.tree_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    
    answer_file = os.path.join(args.answer_dir, args.bench_name, f"{args.model_id}.jsonl")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    print(f"Output to {answer_file}")

    run_eval(
        base_model_path=args.base_model_path,
        ee_model_path=args.ee_model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        datapath=datapath,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
        tree_choices=args.tree_choices,
    )
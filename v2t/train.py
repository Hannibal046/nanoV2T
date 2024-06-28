## built-in
import argparse
import math
import os
import pickle,json
from datetime import timedelta
from functools import partial

## setup ENVIRONMENT VARIABLES
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"]='*.pt' ## not upload ckpt to wandb cloud

## third-party
import nltk
nltk.download('punkt')

### huggingface
import datasets
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate.utils import set_seed
import transformers
from transformers import (
    SchedulerType,
    get_scheduler,
)

### torch
import torch
import torch.utils
import torch.distributed as dist
from tqdm.auto import tqdm

## own
from v2t.utils import write_jsonl,save_with_accelerate,get_yaml_file
from v2t.model import (
    load_generator,
    load_embedder,
)
from v2t.data import load_data,collator
from v2t.overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_train_samples",
        type=int,
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
    )
    parser.add_argument(
        "--workdir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file to launch the training"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        help="stage1 or stage2"
    )
    parser.add_argument(
        "--draft_dir",
        type=str,
        help='directory to store stage1 generated draft'
    )
    parser.add_argument(
        "--dataset_name_or_path", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--generator_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--project_draft_embedding",
        type=eval,
        help="whether to project draft embedding to generator",
    )
    parser.add_argument(
        "--layer_norm_after_projection",
        type=eval,
    )
    parser.add_argument(
        "--expanding_factor",
        type=int,
        help="expand 1 embedding vector to N",
    )
    parser.add_argument(
        "--embedder_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_note", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        "--embedding_construction",
        type=str,
    )
    parser.add_argument(
        "--torch_compile",
        type=eval,
    )
    args = parser.parse_args()
    yaml_config = get_yaml_file(args.config)

    ## priority: CLI > YAML (with all default value set to None in argument parser)
    for k,v in yaml_config.items():
        assert hasattr(args,k), f"{k} not in parsed arguments"
        if getattr(args,k) is None:
            setattr(args,k,v)

    ## Post Init
    if os.path.exists(os.path.join(args.workdir,args.dataset_name_or_path)):
        args.dataset_name_or_path = os.path.join(args.workdir,args.dataset_name_or_path)
    if args.embedder_name_or_path is not None and os.path.isdir(args.embedder_name_or_path):
        args.embedder_name_or_path = os.path.join(args.workdir,args.embedder_name_or_path)
    if os.path.isdir(os.path.join(args.workdir,args.generator_name_or_path)):
        args.generator_name_or_path = os.path.join(args.workdir,args.generator_name_or_path)
    if args.task_type == 'stage2':
        assert args.project_draft_embedding == True

    return args

def display_and_save_one_batch(accelerator,args,batch,tokenizer):
    if accelerator.is_local_main_process:
        pickle.dump(
            batch,
            open(os.path.join(os.path.dirname(args.output_dir),"sample_data.pkl"),'wb'),
        )
    # accelerator.print("**"*20,"show one example","**"*20)
    # accelerator.print("Keys in one batch:==============================================>>")
    # accelerator.print(f"\n {batch.keys()}")
    # if "text" in batch:
    #     accelerator.print("Embedder text:==============================================>>")
    #     accelerator.print(batch['text'][0])
    # if 'labels' in batch:
    #     accelerator.print("Labels:==============================================>>")
    #     accelerator.print(tokenizer.decode(batch['labels'][0]))
    # accelerator.print()
    # accelerator.print('\n'+"**"*20,"show one example","**"*20)

@torch.inference_mode()
def generate_hyps(model,dataloader,accelerator,tokenizer,embedder,max_seq_length):
    
    generation_kwargs = {
        "early_stopping": False,
        "num_beams": 1,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
        "max_length": max_seq_length + 2, ## T5 would add <bos> and <eos>
    }

    model.eval()
    hyps = []
    sub_batch_size = 32
    for batch in tqdm(dataloader,disable= not accelerator.is_local_main_process,desc='Generating...'):

        target_embeddings = batch['target_embeddings']
    
        ## sub-batch-size
        ids_ls = torch.split(torch.tensor(batch['ids']),sub_batch_size)
        target_embeddings_ls = torch.split(target_embeddings, sub_batch_size)

        for sub_tareget_embeddings,ids in zip(target_embeddings_ls,ids_ls):
            generated_ids = accelerator.unwrap_model(model).generate(
                target_embeddings = sub_tareget_embeddings,
                **generation_kwargs,
            )
            generated_hyps = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
            generated_hyps = [
                {"idx":int(idx.item()),"draft":hyp} 
                for idx,hyp in zip(ids,generated_hyps)
            ]
            hyps.extend(generated_hyps)
    
    if accelerator.use_distributed and accelerator.num_processes>1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects,hyps)
        hyps = [x for y in all_ranks_objects for x in y]
    
    ## deduplicate (Data Parallel would pad samples across GPUs)
    seen = set()
    ret = []
    
    for item in hyps:
        idx = item['idx']
        if idx not in seen:
            ret.append(item)
            seen.add(idx)

    return ret

def model_forward(model,batch):
    target_embeddings = batch['target_embeddings']
    
    if 'draft_embeddings' not in batch:
        ## Stage1 forward
        return model(
            input_ids = None,
            target_embeddings = target_embeddings,
            labels = batch['labels'],
        )
    else:
        ## Stage2 forward
        return model(
            input_ids = batch['draft_input_ids'],
            attention_mask = batch['draft_attention_mask'],
            draft_embeddings = batch['draft_embeddings'],
            target_embeddings = target_embeddings,
            labels = batch['labels'],
        )

@torch.inference_mode()
def validate(model,dataloader,accelerator):
    model.eval()
    total_loss = []
    for batch in dataloader:
        outputs = model_forward(model,batch)
        total_loss.append(outputs.loss.item())
    model.train()
    if accelerator.use_distributed and accelerator.num_processes>1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects,total_loss)
        total_loss = [x for y in all_ranks_objects for x in y]
    loss = sum(total_loss)/len(total_loss)
    return loss

def main():
    args = parse_args()
    set_seed(args.seed)

    ## == Init Retriever before Accelerator == ##
    embedder = load_embedder(args.embedder_name_or_path)

    ## == Init Accelerator == ##
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        log_with="wandb",
        kwargs_handlers=[kwargs],
    )
    wandb_dir = os.path.join(args.workdir,"checkpoints",args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(wandb_dir,exist_ok=True)
    accelerator.init_trackers(
        project_name=args.project_name, 
        config=args,
        init_kwargs={
            "wandb": {
                "dir": wandb_dir, 
                "name": args.exp_name,
                "notes": args.exp_note if args.exp_note is not None else None,
                "save_code": True,
            },
        }
    )
    accelerator.print(json.dumps(vars(args),indent=4))
    checkpoint_dir = [None]
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        checkpoint_dir = [str(wandb_tracker.run.dir)]
    if accelerator.use_distributed:dist.broadcast_object_list(checkpoint_dir,src=0)
    args.output_dir = checkpoint_dir[0]
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    embedder = embedder.to(accelerator.device)

    # Make one log on every process with the configuration for debugging.
    overwatch.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()

    ## == Init Generator == ##
    model,tokenizer = load_generator(
        args.generator_name_or_path,
        embedder_name_or_path=args.embedder_name_or_path,
        expanding_factor=args.expanding_factor,
        torch_compile=args.torch_compile,
        project_draft_embedding=args.project_draft_embedding,
        layer_norm_after_projection=args.layer_norm_after_projection,
    )

    ## == Build Dataset == ##
    train_dataset,dev_dataset,test_dataset = load_data(
        args.dataset_name_or_path,
        tokenizer,
        accelerator=accelerator,
        max_train_samples=args.max_train_samples,
        max_seq_length=args.max_seq_length,
        embedding_construction=args.embedding_construction,
        embedder=embedder,
        draft_dir=args.draft_dir,
    )
    collate_fn = partial(collator,tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
        num_workers=8, pin_memory=True,
    )

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False, 
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False, 
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )

    if args.embedding_construction == 'offline':embedder = embedder.cpu()
    
    ## == Build Optimizer == ##
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=1e-6,)

    ## == Calculate Training Configurations == ##
    ### Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # ## == Prepare everything with `accelerator` == ##
    model, optimizer, train_dataloader, lr_scheduler, dev_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler, dev_dataloader, test_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    ## == Train Start! == ##
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    overwatch.info("[bold]Running training[bold]")
    overwatch.info(f"  Dataset = {args.dataset_name_or_path}")
    overwatch.info(f"  Max Sequence Length = {args.max_seq_length}")
    overwatch.info(f"  Generator = {args.generator_name_or_path}")
    overwatch.info(f"  Embedder = {args.embedder_name_or_path}")
    overwatch.info(f"  Num examples = {len(train_dataloader.dataset)}")
    overwatch.info(f"  Num Epochs = {args.num_train_epochs}")
    overwatch.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    overwatch.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    overwatch.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    overwatch.info(f"  Total optimization steps = {args.max_train_steps}")
    overwatch.info(f"  Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)/(10**6):.2f} M") ## not applicable for deepspeed

    ## == init statistics == ##
    best_dev_loss=100
    completed_steps = 0
    logging_interval_loss = 0
    total_loss = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, ncols=100)
    is_first_batch = True
    
    for epoch in range(args.num_train_epochs):
        train_dataloader.set_epoch(epoch)
        model.train()
        for batch in train_dataloader:
            ## == Save and print the first batch data == ##
            if is_first_batch:
                is_first_batch=False
                display_and_save_one_batch(accelerator,args,batch,tokenizer,)

            with accelerator.accumulate(model):
                ## == Model forward == ##
                outputs = model_forward(model,batch)

                ## == Compute Loss == ##
                loss = outputs.loss

                ## == Backprop == ##
                logging_interval_loss += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    norm = accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # === Finish One optimization step == ##
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                ## == Logging == ##
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    total_loss += accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps 

                    to_be_logged = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "rolling_loss":total_loss / completed_steps,
                        "gradient_norm":norm.item(),
                    }
                    progress_bar.set_postfix(dict(loss=avg_loss))
                    accelerator.log(to_be_logged,step=completed_steps)
                    logging_interval_loss = 0
                
                ## == Checkpointint == ##
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        dev_loss = validate(model,dev_dataloader,accelerator)
                        accelerator.log({"dev_loss":dev_loss},step=completed_steps)
                        if dev_loss < best_dev_loss:
                            output_dir = os.path.join(args.output_dir, f"ckpt/best")
                            if accelerator.is_local_main_process:os.makedirs(output_dir,exist_ok=True)
                            save_with_accelerate(accelerator, model, tokenizer, output_dir)
                            best_dev_loss = dev_loss
    

                ## == Early Stop == ##
                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            dev_loss = validate(model,dev_dataloader,accelerator)
            accelerator.log({"dev_loss":dev_loss},step=completed_steps)
            if dev_loss < best_dev_loss:
                output_dir = os.path.join(args.output_dir, f"ckpt/best")
                if accelerator.is_local_main_process:os.makedirs(output_dir,exist_ok=True)
                save_with_accelerate(accelerator, model, tokenizer, output_dir)
                best_dev_loss = dev_loss

    accelerator.wait_for_everyone()
    # generate draft with trained model 
    ckpt_path = os.path.join(args.output_dir, "ckpt/best/ckpt.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(ckpt_path,map_location="cpu"),strict=True)

    ## save
    output_dir = os.path.join(args.output_dir, f"final")
    if accelerator.is_local_main_process: os.makedirs(output_dir,exist_ok=True)
    save_with_accelerate(accelerator, model, tokenizer, output_dir, only_state_dict = False)

    if args.task_type == 'stage1':
        if accelerator.is_local_main_process: os.makedirs(os.path.join(args.output_dir,"hyps"),exist_ok=True)
        for _split in ['train','dev','test']:
            hyps = generate_hyps(
                model,
                dataloader = eval(f"{_split}_dataloader"),
                accelerator=accelerator,
                tokenizer=tokenizer,
                embedder=embedder,
                max_seq_length=args.max_seq_length
            )
            if accelerator.is_local_main_process:
                write_jsonl(hyps,path = os.path.join(args.output_dir,"hyps",f"{_split}.hyps.jsonl"))
    
    accelerator.end_training()

if __name__ == "__main__":
    main()

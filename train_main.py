import torch
import random
import tree_sitter
#parsing programming language syntax tree 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
#batching and shuffling datasets 
from argparse import ArgumentParser
#command-line
from torch.optim.lr_scheduler import ExponentialLR
#learning rate scheduler 

#logs give information about loss rate/accuracy, as well as print out results/training stats 
from logger_setup import setup_logger
from models import (
    # used for training full model (both style and variable)
    # approximates how combined transformation affect code embeddings
    ConcatApproximator, 
    # used when --style_only flag active
    TransformationApproximator,
    # used when --var_only flag active 
    VarApproximator,
    # learn to choose which transformations apply at each step 
    # takes current code embedding and outputs probabiltiy distribution over possible transformation
    # uses vocab mask to ensure only valid transformations selected 
    TransformSelector,
    # Transformer-based encoder that encodes input code tokens into embeddings
    # acts as extractor - used to decode watermark signals
    # learns code syntax and structure 
    TransformerEncoderExtractor,
    # uses Gated Recurrent Units (GRU) to process token sequences
    # when --model_arch gru specified
    # maps sequence of code tokens to embedding vector
    GRUEncoder,
    # separte GRU encoder used when embedding and extraction use different encoderes 
    ExtractGRUEncoder,
    # encodes watermark bits into feature space 
    # takes N_BITS binary bits and maps them to latent vector 
    # learns to inject watermark information into embedding 
    WMLinearEncoder,
    # simple 2-layer MLP (multi-layer percetpron) used as watermark decoder 
    # takes transformed code embeddings as input and tries to recovere embedde bits
    # used with BCE (Binary Cross Entropy) loss to measure reconstruction accuracy 
    MLP2,
)

#custom imports 
from code_transform_provider import CodeTransformProvider #manages trasformations on AST
from runtime_data_manager import InMemoryJitRuntimeDataManager #manages runtime transformation data 
from trainers import UltimateWMTrainer, UltimateVarWMTrainer, UltimateTransformTrainer #contains training loops for experimental setups 
from data_processing import JsonlWMDatasetProcessor, DynamicWMCollator #reads JSONL datasets and converts into tokenized training data 
import mutable_tree.transformers as ast_transformers


#argument parsing - command line options, adds basic arguments 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10) #number of training epochs
    parser.add_argument("--seed", type=int, default=42) #random seed 

    parser.add_argument(
        "--dataset",
        choices=["github_c_funcs", "github_java_funcs", "csn_java", "csn_js"],
        default="github_c_funcs",
    )
    parser.add_argument("--lang", choices=["cpp", "java", "javascript"], default="c")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/github_c_funcs")

    parser.add_argument("--log_prefix", type=str, default="")
    parser.add_argument("--n_bits", type=int, default=4) #bit-width
    #number of bits is size of message we try to watermark in code 
    #more bits moreans more info needs to encode 

    #ablation flags 
    parser.add_argument("--style_only", action="store_true")
    #in keeping with "dual-channel" (change style and variable names)
    parser.add_argument("--var_only", action="store_true")
    parser.add_argument("--var_nomask", action="store_true") # variable channel with no mask (model sees original variable names)
    # masking variable identifiers forces model to predict which name to use 
    #unmasking tests whether masking improves learning 

    parser.add_argument("--varmask_prob", type=float, default=0.5) #probability of masking variable tokens (0.5 = 50% of being masked)
    parser.add_argument("--batch_size", type=int, default=32) #how many samples 
    #weighting factors for variable/style loss components 
    parser.add_argument("--wv", type=float, default=0.5)
    parser.add_argument("--wt", type=float, default=0.5)
    parser.add_argument("--model_arch", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--scheduler", action="store_true") #whether to use learning-rate scheduler 
    parser.add_argument("--shared_encoder", action="store_true") #share encoder between tasks 
    parser.add_argument(
        "--var_transform_mode", choices=["replace", "append"], default="replace"
        #determine how variable transformations applied 
    )

    return parser.parse_args()


def main():
    #main calls above function and stores variable names 
    args = parse_args()
    N_EPOCHS = args.epochs
    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder
    VARMASK_PROB = args.varmask_prob
    VAR_TRANSFORM_MODE = args.var_transform_mode

    logger = setup_logger(args)
    logger.info(args)

    #EDITED THIS 10/23/25
    # device = torch.device("cuda")
    device = torch.device("cpu")
    #normally sets device to use GPU, changed to use CPU for current trial 


    # seed everything
    torch.manual_seed(SEED)
    random.seed(SEED)

    # load datasets
    dataset_processor = JsonlWMDatasetProcessor(lang=LANG)
    # loads .jsonl files, returns dictionary with train/valid/test splits
    logger.info("Processing original dataset")
    instance_dict = dataset_processor.load_jsonls(DATASET_DIR)
    train_instances = instance_dict["train"]
    valid_instances = instance_dict["valid"]
    test_instances = instance_dict["test"]
    # builds vocabulary (token-to-ID mapping)
    vocab = dataset_processor.build_vocab(train_instances)
    # converts examples to torch.utils.data.Dataset objects
    train_dataset = dataset_processor.build_dataset(train_instances, vocab)
    valid_dataset = dataset_processor.build_dataset(valid_instances, vocab)
    test_dataset = dataset_processor.build_dataset(test_instances, vocab)
    # log dataset sizes 
    logger.info(f"Vocab size: {len(vocab)}")
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
    all_instances = train_instances + valid_instances + test_instances

    # initialize transform computers
    # tree-sitter parsing setup: initialize tree-sitter for programming language 
    parser = tree_sitter.Parser()
    parser_lang = tree_sitter.Language("./parser/languages.so", LANG)
    parser.set_language(parser_lang)
    # list of transformations applied to code AST 
    code_transformers = [
        ast_transformers.IfBlockSwapTransformer(),
        ast_transformers.CompoundIfTransformer(),
        ast_transformers.ConditionTransformer(),
        ast_transformers.LoopTransformer(),
        ast_transformers.InfiniteLoopTransformer(),
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer(),
    ]
    # applies transformation data efficiently 
    # CodeTransformProvider - applies language-specific transformations to code
    # LANG (language), parser (TreeSitter), code_transformers (above)
    transform_computer = CodeTransformProvider(LANG, parser, code_transformers)

    # caches and schedules transformation data efficiently, tracks transformation metadat
    # InMemoryJitRuntimDataManager - runtime transformation cache, acts as cache and controller for all transformation data 
    # all_instances - dataset of code samples/ training instances
    transform_manager = InMemoryJitRuntimeDataManager(
        transform_computer, all_instances, lang=LANG
    )
    # setup before training or inference: load vocabulary, transformation feasibility masks, variable name dictionaries, 
    # and compute how much capacity each transformation provides 
    transform_manager.register_vocab(vocab) # register token vocabulary used by model
    # when transformation manager renames variables/modifies code tokens, must ensure new tokens
    # are valid vocab entries and token IDs correspond to model embeddings

    # loads feasibility mask (JSON file) describing which code transformations valid for dataset 
    # ensures transformations are semantics-preserving and language safe 
    transform_manager.load_transform_mask(f"datasets/feasible_transform_{DATASET}.json")

    #loads dictionary of variable names specific to dataset 
    # mapping of possible variable name replacements 
    transform_manager.load_varname_dict(f"datasets/variable_names_{DATASET}.json")

    # queries transform manager for total number of available transformation sites (how many bits can be embedded)
    # transformation capacity = total embeddings in dataset 
    # determined by how many feasible transformations exist (from mask) and which variable names/styles valid
    transform_capacity = transform_manager.get_transform_capacity()

    # creates training, validation, and test data loaders 
    # responsible for feeding batches of transformed code samples (and watermark labels) to model efficiently during training/eval
    logger.info("Constructing dataloaders and models")
    # creates PyTorch DataLoader that wraps dataset, batches data, shuffles if needed
    # uses custom collate function 
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # dynamically generates watermark bits for each sample before batching
        # selects N_BITS watermark bits 
        # prepares input code tokens
        # ataches labels/masks needed for model to learn how to embed/extract those bits 
        collate_fn=DynamicWMCollator(N_BITS),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # shuffle is False (want consistent, repeatable evaluation)
        collate_fn=DynamicWMCollator(N_BITS),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DynamicWMCollator(N_BITS),
    )
    # sets up encoder architecture (GRU or Transformer)
    logger.info(f"Using {MODEL_ARCH}")
    EMBEDDING_DIM = 768 # how large each word/ identifier vector is
    FEATURE_DIM = 768 # hidden state or feature vector size for each encoder 
    if MODEL_ARCH == "gru":
        # processes tokenized code as sequence
        encoder = GRUEncoder(
            # define input dimensions (number of possible tokens)
            vocab_size=len(vocab), hidden_size=FEATURE_DIM, embedding_size=EMBEDDING_DIM
        )
        # can use shared encoder for both encoder and extraction
        # or use two separate encoders 
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            # if not sharing an encoder, instantiate separate ExtractGRUEncoder
            extract_encoder = ExtractGRUEncoder(
                vocab_size=len(vocab),
                hidden_size=FEATURE_DIM,
                embedding_size=EMBEDDING_DIM,
            )
        encoder_lr = 1e-3 # learning rate for GRU
    elif MODEL_ARCH == "transformer":
        encoder = TransformerEncoderExtractor(
            vocab_size=len(vocab), embedding_size=EMBEDDING_DIM, hidden_size=FEATURE_DIM
        )
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = TransformerEncoderExtractor(
                vocab_size=len(vocab),
                embedding_size=EMBEDDING_DIM,
                hidden_size=FEATURE_DIM,
            )
        encoder_lr = 3e-4 # learning rate for Transformer 
        if not args.scheduler:
            encoder_lr = 5e-5
    logger.info(f"learning rate: {encoder_lr}")

    #configures which variable names eligible for transformation, depending on mode
    if VAR_TRANSFORM_MODE == "replace": # replace mode
        # returns boolean mask over vocabulary tokens marking valid variable identifiers 
        # uses to select which tokens can be replace during variable-channel transformations
        vmask = vocab.get_valid_identifier_mask()
    else:
        # instead of all valid identifiers, restrict set to high-frequency identifiers (those that appear often in dataset)
        # scales with N_BITS (number of watermark bits)
        vmask = vocab.get_valid_highfreq_mask(32 * 2**N_BITS)

    # determine which code transformations to apply at each step
    # such as renaming variable, changing identifier, modifying style
    selector = TransformSelector(
        vocab_size=len(vocab), # number of unique tokens (identifiers, keywords, etc) in code vocabulary 
        transform_capacity=transform_capacity, # number of possible transformations availble 
        input_dim=FEATURE_DIM, # size of feature vector input 
        vocab_mask=vmask, # mask indicating which tokens renamed or trasformed 
        random_mask_prob=VARMASK_PROB, # probability of randomly masking variable tokens (variability in learning)
    )
    
    # choose approximator (decide how to combine variable and style information based on ablation flags)
    if args.var_only:
        # used with only variable renaming 
        approximator = VarApproximator(
            vocab_size=len(vocab), input_dim=FEATURE_DIM, output_dim=FEATURE_DIM
        )
    elif args.style_only:
        # used when only style-based transformation active
        approximator = TransformationApproximator(
            transform_capacity=transform_capacity,
            input_dim=FEATURE_DIM,
            output_dim=FEATURE_DIM,
        )
    else:
        # used in full model (both variable and style channels)
        approximator = ConcatApproximator(
            vocab_size=len(vocab),
            transform_capacity=transform_capacity,
            input_dim=FEATURE_DIM,
            output_dim=FEATURE_DIM,
        )
    logger.info(f"approximator arch: {approximator.__class__.__name__}")

    # watermark encoder/decoder
    #encodes watermark bits (binary seq of length N_BITS) into vector in same space as code embeddings (FEATURE_DIM)
    #represents target shift 
    wm_encoder = WMLinearEncoder(N_BITS, embedding_dim=FEATURE_DIM) 
    # decoder - recoers watermark bits from final watermarked code 
    # multi-layer (2-layer network), bn = False means batch normalization disabled
    wm_decoder = MLP2(output_dim=N_BITS, bn=False, input_dim=FEATURE_DIM)

    # moves all neural network modules to compute device (GPU via "cuda" or CPU via "cpu")
    # prepares model components for GPU accelerated training
    encoder.to(device)
    if extract_encoder is not None:
        extract_encoder.to(device)
    selector.to(device)
    approximator.to(device)
    wm_encoder.to(device)
    wm_decoder.to(device)

    # encoder optimizer
    if SHARED_ENCODER:
        # uses AdamW, variant of Adam optimizer 
        # utilizes decoupled weight decay (helps generalization)
        # decoupled weight decay - instead of adding weight decay into gradient, apply directly to weights after gradient step
        # if shared_encoder, then both embeddinga nd extraction share encoder 
        scheduled_optim = optim.AdamW(
            [
                {
                    "params": encoder.parameters(),
                    "lr": encoder_lr,
                    "weight_decay": 0.01,
                },
            ]
        )
    else:
        # otherwise encoder and extract_encoder trained separately
        # same learning rate and weight decay
        scheduled_optim = optim.AdamW(
            [
                {
                    "params": encoder.parameters(),
                    "lr": encoder_lr,
                    "weight_decay": 0.01,
                },
                {
                    "params": extract_encoder.parameters(),
                    "lr": encoder_lr,
                    "weight_decay": 0.01,
                },
            ]
        )
    # other optimizer 
    # uses plain Adam (no weight decay) for all other modules    
    other_optim = optim.Adam(
        [
            {"params": selector.parameters()}, #selector - chooses transformations
            {"params": approximator.parameters()}, # approximator - predicts transformation effects 
            {"params": wm_encoder.parameters()}, # injects watermark
            {"params": wm_decoder.parameters()}, # extracts watermark
        ]
    )
    # optional learning rate scheduler 
    # if scheduler flag enabled, exponential learning rate decay applied
    # gradually reduces learning rate to stabilize training and avoid overshooting once model nears convergence
    if args.scheduler:
        logger.info("Using exponential scheduler")
        scheduler = ExponentialLR(scheduled_optim, gamma=0.85)
    else:
        scheduler = None
    # define loss functions
    # nn.BCELoss is Binary Cross Entropy Loss
    # used when model outputs probability between 0 and 1 
    # loss measures how  close predicted watermark bits are to true bits
    loss_fn = nn.BCELoss()

    # starts training 
    logger.info("Starting training loop")
    # builds checkpoint/log directory name 
    save_dir_name = f"{args.log_prefix}_{args.seed}_{args.dataset}"
    # style only training 
    if args.style_only:
        logger.info("[ABLATION] Style only")
        # UltimateTransformTrainer - trains style-only watermarking, no variable renaming
        trainer = UltimateTransformTrainer(
            code_encoder=encoder,
            extract_encoder=extract_encoder,
            wm_encoder=wm_encoder,
            selector=selector,
            approximator=approximator,
            wm_decoder=wm_decoder,
            scheduled_optimizer=scheduled_optim,
            other_optimizer=other_optim,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            transform_manager=transform_manager,
            scheduler=scheduler,
            logger=logger,
            ckpt_dir=save_dir_name,
        )
    elif args.var_only:
        # UltimateVarWMTrainer - variable only 
        logger.info("[ABLATION] Var only")
        trainer = UltimateVarWMTrainer(
            code_encoder=encoder,
            extract_encoder=extract_encoder,
            wm_encoder=wm_encoder,
            selector=selector,
            approximator=approximator,
            wm_decoder=wm_decoder,
            scheduled_optimizer=scheduled_optim,
            other_optimizer=other_optim,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            transform_manager=transform_manager,
            scheduler=scheduler,
            logger=logger,
            ckpt_dir=save_dir_name,
        )
    else:
        # full mode - no ablation
        trainer = UltimateWMTrainer(
            encoder,
            extract_encoder,
            wm_encoder,
            selector,
            approximator,
            wm_decoder,
            scheduled_optim,
            other_optim,
            device,
            train_loader,
            valid_loader,
            test_loader,
            loss_fn,
            transform_manager,
            w_var=args.wv,
            w_style=args.wt,
            scheduler=scheduler,
            logger=logger,
            ckpt_dir=save_dir_name,
            var_transform_mode=VAR_TRANSFORM_MODE,
        )
        # mask configuration 
        # if --var_nomask not specified, random masking enabled
        trainer.set_var_random_mask_enabled(not args.var_nomask)
        logger.info(f"w_var: {args.wv}, w_style: {args.wt}")
        logger.info(f"var mask enabled: {not args.var_nomask}")

    trainer.do_train(N_EPOCHS)


if __name__ == "__main__":
    main()

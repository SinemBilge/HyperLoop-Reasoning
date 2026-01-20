from math import exp, log
class BaseTrainingConfig:
        def __init__(self, optimizer = 'AdaFactor',
                     learning_rate = 0.001,
                     optimizer_param = 1e-5,
                     curvature = 1.0,
                     epochs = 50,
                     scheduler = None,
                     scheduler_factor = 1,
                     log_dir = 'default_log',
                     model_save_path = 'default_model_save',
                     tboard_checkpoint_path = None,
                     model_checkpoint_path = None,
                     load_optimizer = False,
                     additional_log_info = '',
                     num_workers = 4,
                     gpu_parallelization = False):
            self.optimizer = optimizer #[Adam, AdamW, AdaFactor, Hyperbolic]
            self.learning_rate = learning_rate #Same as in the paper
            self.optimizer_param = optimizer_param #weight_decay for AdamW Check which Optimizer
            self.epochs = epochs # 100K steps with early Stopping 
            self.scheduler = scheduler  
            self.scheduler_factor = scheduler_factor
            self.log_dir = log_dir
            self.model_save_path = model_save_path
            self.tboard_checkpoint_path = tboard_checkpoint_path
            self.model_checkpoint_path = model_checkpoint_path
            self.load_optimizer = load_optimizer
            self.num_workers = num_workers
            self.curvature = curvature
            self.additional_log_info = additional_log_info
            self.gpu_parallelization = gpu_parallelization
            
class Config:  
    class SingleHopTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(log_dir='tboard_logs/metaqa/knowledge_integration',
                             model_save_path='checkpoints/metaqa/knowledge_integration',
                             model_checkpoint_path= None,
                             tboard_checkpoint_path=None,
                             scheduler=None,
                             num_workers=0,
                             curvature=log(exp(0.1) - 1),
                             gpu_parallelization=True,  # Set to False for Mac/CPU training
                             learning_rate=0.001,
                             epochs=50)
            self.additional_log_info=f'knowledge_integration_bsize64_lr0.001'

                
    class OneHopWikiTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(log_dir='tboard_logs/metaqa/one_hop_wiki_finetuning',
                             model_save_path='checkpoints/metaqa/one_hop_wiki_finetuning',
                             epochs=160,
                             model_checkpoint_path= None,
                             tboard_checkpoint_path=None,
                             num_workers=16)

    class RandomWalkTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(learning_rate=0.3,
                             epochs=100,
                             log_dir='tboard_logs/metaqa/random_walk_training/euclidean',
                             model_save_path='checkpoints/metaqa/random_walk_training/euclidean', #Dec27_09-23-05_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_scheduler_only_single_answer_kb_hyperbolic/knit5.pth #Dec26_14-35-36_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_scheduler_only_single_answer_kb/knit5.pth
                             model_checkpoint_path= 'checkpoints/metaqa/knowledge_integration/Dec26_14-35-36_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_scheduler_only_single_answer_kb/knit5.pth', 
                             tboard_checkpoint_path=None,
                             num_workers=1,
                             optimizer='AdaFactor',
                             curvature=log(exp(0.32) - 1),
                             gpu_parallelization=True
                             )
            self.use_scheduler = False
            self.use_soft_prompt = True
            self.prompt_length = 100
            self.additional_log_info=f'identity_after_encoder_bsize64_prompt_lenght{self.prompt_length}_lr{self.learning_rate}_curvature{self.curvature}_additional_layer_lr0.001_max_answer_2'
            self.hopping_prompt_checkpoint_path = None

    class ParseThenHopTraining(BaseTrainingConfig):
        def __init__(self):
            super().__init__(learning_rate=0.8,
                             epochs=250,
                             log_dir='tboard_logs/metaqa/parse_training/',
                             model_save_path='checkpoints/metaqa/parse_training/',
                             model_checkpoint_path= 'checkpoints/musique_dataset/knowledge_integration/euclidean_gt_not_replaced/knit5_epoch_28_val_loss_0.0045.pth',
                             tboard_checkpoint_path=None,
                             num_workers=0,
                             curvature=log(exp(1.0) - 1),
                             gpu_parallelization=True
                             )
            self.use_scheduler = False
            self.use_soft_prompt = True
            self.prompt_length = 100
            self.additional_log_info=f'parse_training_gt_not_replaced_euclidean_linear_layer_lr{self.learning_rate}_bsize16'
            self.hopping_prompt_checkpoint_path = None
            self.parsing_prompt_checkpoint_path = None
            
                
    class T5_Model:
        def __init__(self):
            self.batch_size = 64
            self.model_name = "google/t5-large-lm-adapt"            
            self.tokenizer_max_length = 512
            self.map_encoder_layers = []
            self.map_decoder_layers = []


            
    def __init__(self):
        self.t5_model = self.T5_Model()
        self.one_hop_wiki_training = self.OneHopWikiTraining()
        self.single_hop_training = self.SingleHopTraining()
        self.random_walk_training = self.RandomWalkTraining()
        self.parse_then_hop_training = self.ParseThenHopTraining()

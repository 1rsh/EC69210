from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
import random
import yaml

class RunConfig:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            content = yaml.safe_load(file)
        self.__dict__.update(content)
    
    def extract(self):
        BATCH_SIZE = self.train['batch_size']
        NUM_EPOCHS = self.train['num_epochs']
        LEARNING_RATE = self.train['learning_rate']
        DEVICE = self.train['device']
        OPTIMIZER = self.train['optimizer']
        GRAD_CLIP = self.train['grad_clip']
        GRAD_ACCUM_STEPS = self.train['grad_accum_steps']
        NUM_CLASSES = self.model['num_classes']
        PRINT_EVERY = self.logging['print_every']
        SAVE_MODEL = self.logging['save_model']
        MODEL_SAVE_PATH = self.logging['model_save_path']
        return BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE, OPTIMIZER, GRAD_CLIP, GRAD_ACCUM_STEPS, NUM_CLASSES, PRINT_EVERY, SAVE_MODEL, MODEL_SAVE_PATH
    
    def __repr__(self):
        return yaml.dump(self.__dict__)
    

class MyLogger:
    def __init__(self):
        self.cached_df = None

    def log_metrics(self, dictionary, index_key="epoch"):
        clear_output(wait=True)
        
        if self.cached_df is not None:
            oncoming = pd.DataFrame.from_records([dictionary])
            if index_key in oncoming.columns:
                oncoming = oncoming.set_index(index_key)
            self.cached_df = pd.concat([self.cached_df, oncoming])
        else:
            self.cached_df = pd.DataFrame.from_records([dictionary])

        if index_key in self.cached_df.columns:
            self.cached_df = self.cached_df.set_index(index_key)
        display(self.cached_df)
    
    def plot_run(self, keys=None, log=False):
        if keys is None:
            keys = self.cached_df.columns

        for key in keys:
            plt.plot(self.cached_df[key], label=key)
        
        if log:
            plt.yscale("log")   
        plt.legend()
        plt.show()

def show_image_samples(dataset, idx2class=None):
    sampled = random.sample(range(len(dataset)), 9)
    sampled = [(dataset[i][0], dataset[i][1]) for i in sampled]
    fig, axs = plt.subplots(3, 3)
    plt.suptitle('Sample Images')
    for i, (img, label) in enumerate(sampled):
        ax = axs[i // 3, i % 3]
        ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
        if idx2class:
            label = idx2class[label]
        ax.set_title(label)
        ax.axis('off')

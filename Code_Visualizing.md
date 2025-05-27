### trainer.py - Class Trainer
#### the calling trajectory of Trainer.train()
``` python
Class partial:
    def __call__():
        return self.func

def find_executable_batch_size():
    return partial(recurs the function find_executable _batch_size() to search for the best batch size)

Class Triner
    def train():
        inner_training_loop = find_executable_batch_size(func = self._inner_training_loop)
        return inner_training_loop  # call the object partial witch will run the function self._inner_training_loop()

    def self._inner_training_loop():
```
#### the structure of function Train.trainer()
``` python
def train():
    set the batch size
    load the batched data
    set the num_epochs or max_steps
```
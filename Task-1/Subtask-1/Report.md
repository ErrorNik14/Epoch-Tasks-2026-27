# Subtask-1:   Sorted Relative Ranks

This report contains information about my attempts at implementing the Transformer Encoder architecture (and the baselines!)
I have also chosen this sub-task for performing my ablations and analysing the model's behaviours with respect to many parameters/architecture motivations.

## LLM usage
I have commented next to LLM-generated code lines specifically in the source files. LLMs were used to understand concepts like  An LLM was used to generate the diagrams given below. 

## Methodolgy
I decided to make three baselines-
    1. MLP-based model
    2. Vanilla Uni-Directional model
    3. LSTM-based Bidirectional model

After which, I implemented the Transformer Encoder architecture, which I will refer to by 'BERT' for simplicity...\
Given below are diagrams of the architectures used.

### 1. MLP Architecture
<img width="450" height="400" alt="mlp_arch" src="https://github.com/user-attachments/assets/f200275a-a92a-4654-8b68-5752b9e0bf68" />

### 2. RNN Architecture
<img width="450" height="400" alt="rnn_arch" src="https://github.com/user-attachments/assets/68e15680-0295-495f-be7b-f821b126590f" />

### 3. biLSTM Architecture
<img width="450" height="400" alt="lstm_arch" src="https://github.com/user-attachments/assets/612bcf6f-8985-4935-b717-6410a8fdecb6" />

### 3. "BERT" Architecture
<img width="450" height="400" alt="bert_arch" src="https://github.com/user-attachments/assets/08a4464e-b493-4ff5-a2cc-0f10972ebbce" />

I wanted to gauge the extent to which these models could handle varying length. So I took the raw dataset provided, and sliced them into different lengths (4, 6, 8, and the original 10), creating datasets containing sequences and ranks of lengths 4 to 10. I plan on using these to understand the above.

I have also made my source code modular enough to toggle embeddings and/or normalisations on and off, though I've left this out for the "BERT" model, as I feel it isn't useful for such a complex model to be trained on non-embedded data. 

With that being said, the baseline parameters of each model for the initial evaluations will be as follows...  
* <b>MLP</b> (`vocab_size=1000`, `sequence_lengths=[4,6,8,10]`, `embed_dim=5`)   
* <b>Vanilla RNN</b> (`vocab_size=1000`, `sequence_lengths=[4,6,8,10]`, `embed_dim=5`, `layer=1`, `bidirectional=False`)
* <b>biLSTM</b> (`vocab_size=1000`, `sequence_lengths=[4,6,8,10]`, `embed_dim=5`, `layer=1`, `bidirectional=True`)
* <b>BERT</b> (`N=3`, `heads=2`, `dmodel=10`, `dk=5`, `dv=5`, `vocab_size=1000`, `sequence_lengths=[4,6,8,10]`)  

with an AdamW optimizer (`lr=1e-3`, `weight_decay=1e-3`, `betas=(0.9,0.99)`), and a batch_size=256. Also, I will let the `maximum number of epochs` be `2000`. All them will be tested with embeddings initially (which also means no normalisation), with more work done in ablations.  
For testing metrics, I will be using `CrossEntropyLoss` and `Element-wise Matches`.

## Baseline Analysis
Following are the training-validation loss history of the models, for sequence_length=10.  
MLP Model &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;RNN Model &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;biLSTM Model  
<img width="300" height="250" alt="10_mlp_loss" src="https://github.com/user-attachments/assets/998166aa-6f79-4e6b-be5b-f58d78845d7f" />
<img width="300" height="250" alt="10_rnn_loss" src="https://github.com/user-attachments/assets/d17da529-cb34-4cee-9248-f67f4b885d17" />
<img width="300" height="250" alt="10_lstm_loss" src="https://github.com/user-attachments/assets/434a515f-3557-4bfa-b6f3-b1bd8320ac20" />  
Testing loss= 1.354&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Testing loss= 1.566&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Testing loss= 0.913  
Average matches= 4.849&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Average matches= 3.536&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Average matches= 6.057  
  
We can interpret these results well already.  
      1. MLP model, the most basic of the models used, is able to train stably without much overfitting, due to it's simplicity. But it only considers inputs in parallel ; It lacks the ability to tell apart the positions of elements, thus it shows poor inference.  
      2. Vanilla RNN model is able to account for the sequential nature of the data, but it is prone to "forgetting" information about the previous tokens in its hidden state, and performs very poorly. For the purpose of this sub-task, I chose a Unidirectional RNN network, and thus it suffers even more.  
      3. biLSTM is much better at keeping track of earlier tokens due to its hiddenstate-cellstate architecture, and Forget-Input-Output gates.  

Let's dive a little deeper into some specifics.
### Training Convergence
Owing to the order of complexities, MLP < Vanilla RNN < biLSTM is the order in which they converge to the lowest validation loss. Though we can note how Vanilla RNN and biLSTMs being more complex, have a tendancy to overfit much faster, as they converge first. They are trickier to train and suitable dropout was essential.

### Validation Loss
MLPs, as highlighted earlier, are unable to gauge the order of elements in the sequence. Vanilla RNNs are not good at retaining important information in their hidden states and also suffer in terms of validation performance. Only a bidirectional-LSTM is able to overcome both issues, with the added advantage that it's able to compare elements from both ends of the sequence. (Although a bidirectional-GRU could have also been used.)  

### Stability
All three models display smooth descent in loss history, but Vanilla RNNs and biLSTMs drop to their loss minimas quickly and wildly start overfitting. The MLP model is able maintain a healthy gap between validation and training loss at the plateau region - signalling that for the architecture, it is generalising well and not overfitting. Same cannot be said about the other two models... 

### Generalization Behavior
Observe some testing examples from each of the models.
1. MLP model  
&emsp;&emsp;&emsp;&emsp;Sequence-->[ 42, 954,  20, 706, 244, 717, 279, 440, 350, 439]  
&emsp;&emsp;&emsp;&emsp;Predic-->   [0, 7, 0, 7, 2, 7, 3, 5, 4, 5]   
&emsp;&emsp;&emsp;&emsp;Actual-->   [1, 9, 0, 7, 2, 8, 3, 6, 4, 5]  

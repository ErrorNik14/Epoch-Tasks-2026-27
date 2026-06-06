# Subtask-1:   Sorted Relative Ranks

This report contains information about my attempts at implementing the Transformer Encoder architecture (and the baselines!)
I have also chosen this sub-task for performing my ablations and analysing the model's behaviours with respect to many parameters/architecture motivations.

## LLM usage
I have commented next to LLM-generated code lines specifically in the source files. LLMs were used to debug CUDA errors and other operational faults. An LLM was used to generate the diagrams and table structures used across, but not the values in them. 

## Methodology
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
* <b>BERT</b> (`N=3`, `heads=2`, `dmodel=10`, `dk=5`, `dv=5`, `vocab_size=1000`, `sequence_lengths=[4,6,8,10]`) (with Xavier initialisation)

with an AdamW optimizer (`lr=1e-3`, `weight_decay=1e-3`, `betas=(0.9,0.99)`), and a batch_size=256. Training-Validation-Test split of 80-10-10. Also, I will let the `maximum number of epochs` be `2000`. All them will be tested with embeddings initially (which also means no normalisation), with more work done in ablations.  
For testing metrics, I will be using `CrossEntropyLoss` and `Element-wise Matches`.

## Baseline Analysis
<table>
  <tr>
    <td align="center" width="300">MLP Model</td>
    <td align="center" width="300">Vanilla RNN Model</td>
    <td align="center" width="300">biLSTM Model</td>
  </tr>
  <tr>
    <td><img width="300" height="250" src="https://github.com/user-attachments/assets/998166aa-6f79-4e6b-be5b-f58d78845d7f"/></td>
    <td><img width="300" height="250" src="https://github.com/user-attachments/assets/d17da529-cb34-4cee-9248-f67f4b885d17"/></td>
    <td><img width="300" height="250" src="https://github.com/user-attachments/assets/434a515f-3557-4bfa-b6f3-b1bd8320ac20"/></td>
  </tr>
  <tr>
    <td align="center">Testing loss = 1.354<br>Average matches = 4.849</td>
    <td align="center">Testing loss = 1.566<br>Average matches = 3.536</td>
    <td align="center">Testing loss = 0.913<br>Average matches = 6.057</td>
  </tr>
</table>
  
We can interpret these results well already.  
      1. MLP model, the most basic of the models used, is able to train stably without much overfitting, due to it's simplicity. But it only considers inputs in parallel ; It lacks the ability to tell apart the positions of elements, thus it shows poor inference.  
      2. Vanilla RNN model is able to account for the sequential nature of the data, but it is prone to "forgetting" information about the previous tokens in its hidden state, and performs very poorly. For the purpose of this sub-task, I chose a Unidirectional RNN network, and thus it suffers even more.  
      3. biLSTM is much better at keeping track of earlier tokens due to its hiddenstate-cellstate architecture, and Forget-Input-Output gates.  

Let's dive a little deeper into some specifics.
### Training Convergence
Owing to the order of complexities, MLP < Vanilla RNN < biLSTM is the order in which they converge to the lowest validation loss. Though we can note how Vanilla RNN and biLSTMs being more complex, have a tendency to overfit much faster, as they converge first. They are trickier to train and suitable dropout was essential.

### Validation Loss
MLPs, as highlighted earlier, are unable to gauge the order of elements in the sequence. Vanilla RNNs are not good at retaining important information in their hidden states and also suffer in terms of validation performance. Only a bidirectional-LSTM is able to overcome both issues, with the added advantage that it's able to compare elements from both ends of the sequence. (Although a bidirectional-GRU could have also been used.)  

### Stability
All three models display smooth descent in loss history, but Vanilla RNNs and biLSTMs drop to their loss minimas quickly and wildly start overfitting. The MLP model is able maintain a healthy gap between validation and training loss at the plateau region - signalling that for the architecture, it is generalising well and not overfitting. Same cannot be said about the other two models... 

### Generalization Behavior  
Observe some testing examples from each of the models.  
1. MLP model  
&emsp;&emsp;&emsp;&emsp;Sequence   --> [ 42, 954,  20, 706, 244, 717, 279, 440, 350, 439]  
&emsp;&emsp;&emsp;&emsp;Predicted-->   [0, 7, 0, 7, 2, 7, 3, 5, 4, 5]   
&emsp;&emsp;&emsp;&emsp;Actual   -->   &emsp;[1, 9, 0, 7, 2, 8, 3, 6, 4, 5]  
`Number of matches = 6`, The model does a decent job - it gets 5/10 of the ranks right. But it repeats ranks - notices how it directly assigns lower ranks to smaller numbers, and higher ranks for much larger numbers? `rank 0` for `42` and `20`, while it gives rank 7 for `954`, `706`, `717`? Simply because it doesn't see the numbers as a sequence, but as an independent collection. It's almost vaguely ranking the numbers into ranks based on all 1000 numbers in the vocabulary!  

2. Vanilla RNN Model  
&emsp;&emsp;&emsp;&emsp;Sequence   --> [ 24, 969, 186,  25, 551, 613, 792, 247, 262, 856]  
&emsp;&emsp;&emsp;&emsp;Predicted-->   [0, 9, 1, 3, 5, 5, 8, 6, 4, 6]  
&emsp;&emsp;&emsp;&emsp;Actual   -->   &emsp;[0, 9, 2, 1, 5, 6, 7, 3, 4, 8]  
`Number of matches = 4`, Again it is doing something similar to what the MLP model did - repeating ranks, and <i>attempting</i> broad statistics, like `rank 0` for `24`, `rank 8` for 792, `rank 9` for 969, but it also gets confused because of the partial sequential data it remembers, like how it gives `rank 4` and `rank 6` for `262` and `856` - Neither capturing global statistics in the vocabulary like the MLP, neither capturing global sequence information, thus performing even worse than the MLP model!  

3. biLSTM Model  
&emsp;&emsp;&emsp;&emsp;Sequence   --> [524, 825,  88, 616, 266, 641, 380, 992,  11, 163]   
&emsp;&emsp;&emsp;&emsp;Predicted-->   [5, 8, 1, 6, 2, 6, 4, 9, 0, 2]  
&emsp;&emsp;&emsp;&emsp;Actual   -->   &emsp;[5, 8, 1, 6, 3, 7, 4, 9, 0, 2]  
`Number of matches = 8`, This model is able to actually process sequential information without forgetting it. It's only fault is that in this example, it repeated `rank 2` and `rank 6` twice - but it can just be a fault in the training, since the actual values are pretty close (`rank 3` and `rank 7`).  

### Varying Sequence Length
Now let's observe what happens to the baseline models from a sequence length of 4 to 10, instead of just the current 10. Do these models perform better in smaller sequences? If yes, this is a severe bottleneck.
<table>
<tr>
<td>

**Testing Loss**

| Model | seq=4 | seq=6 | seq=8 | seq=10 |
|:------|:-----:|:-----:|:-----:|:------:|
| MLP | 0.404 | 0.746 | 1.095 | 1.354 |
| Vanilla RNN | 0.662 | 1.050 | 1.337 | 1.566 |
| biLSTM | 0.251 | 0.594 | 0.666 | 0.913 |

</td>
<td>

**Average Matches**

| Model | seq=4 | seq=6 | seq=8 | seq=10 |
|:------|:-----:|:-----:|:-----:|:------:|
| MLP | 3.860 | 4.359 | 4.829 | 4.849 |
| Vanilla RNN | 2.755 | 3.166 | 3.387 | 3.536 |
| biLSTM | 3.633 | 4.430 | 5.717 | 6.057 |

</td>
</tr>
</table>
Oof. This is a shortcoming of all the baseline models.  
In Vanilla RNN, it is able to get 2.755/4 position ranks correct on average, but as we increase it to 10, it hits a ceiling of 3.536/10 correct position ranks.  
In MLP, from 3.86/4 to 4.85/10 correct position ranks, and in biLSTMs, from 3.63/4 to 6.057/10.  
Especially for the RNN/LSTM models, this doesn't seem very prospective to use them, considering they are more expensive to train. Speaking of which...

### Computational Cost
We can gauge this by just recalling the computational complexities of the three models.
| Model | Complexity |
|:------|:----------:|
| MLP | $O(ih + hk)$ |
| Vanilla RNN | $O(ih + h^2 + hk)$ |
| biLSTM | $O(ih + h^2 + h + hk)$ |

(where i = input size, h = hidden size, k = output size)
In this regard for even a well-optimized system of models, MLP > Vanilla RNN > biLSTM in terms of training efficiency.

## Transformer Encoders - A brief analysis
Moving onto the main model of importance - a Transformer Encoder, a model that utilizes self-attention to make sense of the data.
Let us first look at its training and performance on the 10-elements long sequence data set.
Given below is the loss history...  
<img width="300" height="250" alt="10_bert_loss" src="https://github.com/user-attachments/assets/505b2e15-0885-44a3-924b-d1f51898e5e8" />  
With a `Testing loss of 0.903` and an `Average match of 8.586`. It is a much slower to converge model, and requires much more regularisation than the others due to its complexity relative to the sub-task. Nonetheless, it generalises better than even the biLSTM. Infact, it is even able to perfectly predict a significant number of sequences in inference. For example,  
&emsp;&emsp;&emsp;&emsp;Sequence   --> [511, 896,  95, 259, 304, 440, 782, 708, 156, 661]    
&emsp;&emsp;&emsp;&emsp;Predicted-->   [5, 9, 0, 2, 3, 4, 8, 7, 1, 6]    
&emsp;&emsp;&emsp;&emsp;Actual   -->   &emsp;[5, 9, 0, 2, 3, 4, 8, 7, 1, 6]  

It is able to do this because of the bidirectional Self-Attention mechanism baked into it. It is able to attend to every other token with respect to a token, and compare everything globally. 

Further, it is just as good at smaller sequence lengths.  
| Metric | seq=4 | seq=6 | seq=8 | seq=10 |
|:-------|:-----:|:-----:|:-----:|:------:|
| Testing loss | 0.514 | 0.884 | 0.910 | 0.903 |
| Avg matches | 3.554 | 3.958 | 6.396 | 8.586 |  

## Head-wise Attention Visualisation
Let's observe how the model attends to each token, using the `bertviz` library's `head_view()` function.  
<img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/4013b994-f4e1-4a08-a5b5-cffc692cf45d" />  
This is the attention visualisation for all tokens to eachother. The two colours represent the attention from each of the two heads.  
We shall observe some examples and see what emergent properties have arisen.
* Layer 0 - `head2` seems to have learnt rank-wise adjacent elements. Most tokens attend to the rank-wise neighbours, showing signs of pair-wise comparisons and global comparisons. `head1` remains mostly spread out.  
   <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/38bf829c-3349-4886-bbd5-3ee61299c489" /> <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/e8a1ac5a-4975-4902-97e4-e1bbe45e57f4" /> <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/de4583d9-312f-4519-ad94-efd0c3a84c3a" />  
* Layer 1 - Similarly, `head1`-only properties are visible. Most tokens either attend to the rank-wise smaller or greater token. The rank-wise greatest element only attends to itself. Now, `head2` remains almost silent for all the tokens.  
<img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/1d4ec509-05bc-4878-a2e2-2980563fdfe4" /> <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/fb5c0e0b-b923-40c3-b8e3-3337cda56de4" /> <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/4d8dfb5a-6135-4f8c-b02c-36216a8f96a6" />
* Layer 2 - Both `head1` and `head2` are active - Almost all tokens seem to be attending to both rank-wise adjacent tokens, except the smallest and largest token, who attend to the rank-wise larger and smaller token, respectively. The attention given by `head1` is seemingly stronger than that by `head2`.  
<img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/73d8fdf9-55ab-4197-b498-03dedf0b6f6b" /> <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/e5d563f8-1f9d-4cee-921b-afab97e5384a" /> <img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/c10163ae-4aa2-4548-8494-97619749492d" />  
  
  
It is clear from these that the "BERT" model has indeed learnt to-  
* split learning between the heads over the layers, some capturing rank-wise next or previous, while some capture rank-wise locality.  
* compare elements pair-wise.  
* understand the concept of a minimum element and a maximum element.  
* sorting globally across the sequence.  

## Ablation Studies
Now moving onto my ablation studies. They will be divided into two segments-  
* Data Representation ablations - checking the usability of different modes of representing the sequences with a particular baseline model.
* Architecture & Attention ablations - direct comparisons of baselines and self attention model, self attention model and variating architecture / hyper parameters. 

### Ablation-A : Data-Representation  
I will be using the biLSTM baseline with a `sequence length=6`, and an optimizer `AdamW(lr=1e-3,weight_decay=1e-3,betas=(0.9,0.99)))`.  
  
#### Sequence Normalization  
We will be exploring three normalisation strategies-  
    1. Z-Normalisation  
    2. Global Min-Max Normalisation  
    3. Sequence-wise Min-Max Normalisation  

<table>
  <tr>
    <td align="center" width="300">No Normalisation</td>
    <td align="center" width="300">Z-Normalisation</td>
    <td align="center" width="300">Global Min-Max Normalisation</td>
    <td align="center" width="300">Sequence-wise Min-Max Normalisation</td>
  </tr>
  <tr>
    <td><img width="300" height="250" alt="rawnumerical_6_lstm_loss" src="https://github.com/user-attachments/assets/08cf3415-ad09-42be-9052-54a96ae612bf" /></td> 
    <td><img width="300" height="250" alt="norm1_6_lstm_loss" src="https://github.com/user-attachments/assets/69bbf1d2-3390-4e21-a92f-b9bb1a531993" /></td> 
    <td><img width="300" height="250" alt="norm2_6_lstm_loss" src="https://github.com/user-attachments/assets/3698d62f-85db-4688-a51b-d70ff141c91a" /></td>
    <td><img width="300" height="250" alt="norm3_6_lstm_loss" src="https://github.com/user-attachments/assets/f5a65868-811d-4179-bd26-5ede2a1436c7" /></td>
  </tr>
</table>  

* <b>Training stability</b> - No normalisation offers rough descent, plateauing chaotically. Z-Normalisation offers the highest training stability of the four options. Global Min-Max scaling offers the worst performance - the model loses all information as very tiny values get compressed even more. Sequence-wise Min-Max scaling has a mostly smooth descent, placing it close to the best.  
  
* <b>Convergence</b> - No normalisation has chaotic convergence, it gets stuck in a local gradient and starts over-fitting. Sequence-wise Min-Max scaling and Z-normalisation offer superior results, while Global Min-Max scaling leads to chaos and no proper convergence.  
  
* <b>Out-of-distribution performance</b> - Upon testing with an example, `[1, 2, 3, 4, 5, 6]`  
  No normalisation ---> `[4, 1, 4, 4, 2, 2]`  
  Z-normalisation ---> `[0, 0, 0, 1, 5, 5]`  
  Global Min-Max scaling ---> `[5, 5, 5, 5, 4, 5]`  
  Sequence-wise Min-Max scaling ---> `[0, 0, 0, 1, 2, 5]`
    
  Or for an example `[100, 200, 300, 400, 500, 600]`,  
  No normalisation ---> `[0, 1, 2, 4, 4, 5]`  
  Z-normalisation ---> `[0, 0, 1, 4, 5, 1]`  
  Global Min-Max scaling ---> `[5, 0, 5, 5, 0, 5]`  
  Sequence-wise Min-Max scaling ---> `[0, 2, 5, 3, 5, 2]`  
  It <i>can</i> be argued that Z-Normalisation and Sequence-wise Min-Max scaling are much better than the other options, as they do seem to rank outliers somewhat   well.  

#### Categorical Embeddings vs Continuous Representations  
We will be exploring three representation strategies-  
    1. Raw float inputs  
    2. Normalised float inputs (Z-normalisation)  
    3. Embedded inputs  
<table>
  <tr>
    <td align="center" width="300">Raw float inputs</td>
    <td align="center" width="300">Normalised inputs</td>
    <td align="center" width="300">Embedded inputs Normalisation</td>
  </tr>
  <tr>
    <td><img width="300" height="250" alt="rawnumerical_6_lstm_loss" src="https://github.com/user-attachments/assets/08cf3415-ad09-42be-9052-54a96ae612bf" /></td> 
    <td><img width="300" height="250" alt="norm1_6_lstm_loss" src="https://github.com/user-attachments/assets/69bbf1d2-3390-4e21-a92f-b9bb1a531993" /></td> 
    <td><img width="300" height="250" alt="embed_6_lstm_loss" src="https://github.com/user-attachments/assets/6105ea45-1812-4310-a0c2-740519fa6701" /></td> 
  </tr>
</table>  
  
* <b>Training stability and Convergence</b> - Raw float inputs offer rough descent, and plateaus chaotically. Normalised inputs offer the highest training stability of the three options. Embedded inputs offers the worst performance - smooth descent initially but overfits after hitting a local gradient. This can be because the inputs just don't have enough high dimensional information needed to be represented effectively by embeddings.  
  
### Ablation-B : Architecture & Attention  
#### Summarising Baselines vs Transformer  
* <b>Sequential processing</b> — Vanilla RNN / LSTM process tokens one at a time, naturally suited for short sequences. But as sequence length grows, both struggle — the LSTM hits a ceiling of 6.057/10 matches at seq=10, while the Transformer reaches 8.586/10. This is a consequence of information getting compressed into hidden states and/or cell states. Attention mitigates this.  
  
* <b>Long-range reasoning</b> — Vanilla RNNs degrade over long sequences due to vanishing gradients; LSTMs mitigate this with having Input/Output/Forget/Cell_state gates, but still fall short. The BERT" model's bidirectional self-attention attends to all tokens simultaneously, giving it a clear edge as sequence length increases.  
  
* <b>Parallelism</b> — Vanilla RNN / LSTM are inherently sequential — each timestep depends on the previous, making them slow to train. Transformers compute all attention scores in parallel, making them significantly faster on hardware like GPUs.  
  
* <b>Global context understanding</b> — RNN/LSTM build context incrementally and can lose earlier information. The "BERT" model attends to the entire input at once, which is why it learns pairwise comparisons and global ranking structure — as seen in the attention visualisations previously.
    
#### Encoder Depth Experiments  
For this ablations, I will be using the "BERT" baseline with `batch_size=512`, `BERTModel(N=[1,2,3,4], h=2, dmodel=10, dk=5, dv=5, vocab_size=1000, seq_len=6)`, and an optimizer `AdamW(lr=1e-3,weight_decay=1e-3,betas=(0.9,0.99)))`.  
<table>
  <tr>
    <td align="center" width="300">N=1</td>
    <td align="center" width="300">N=2</td>
    <td align="center" width="300">N=3</td>
    <td align="center" width="300">N=4</td>
  </tr>
  <tr>
    <td><img width="300" height="250" alt="N1_6_bert_loss" src="https://github.com/user-attachments/assets/bbc24c2c-4e5e-4b29-8fae-764e5c1e45bc" /></td> 
    <td><img width="300" height="250" alt="N2_6_bert_loss" src="https://github.com/user-attachments/assets/fbed3b65-850e-49cf-ae38-3faf1ccb8295" /></td> 
    <td><img width="300" height="250" alt="N3_6_bert_loss" src="https://github.com/user-attachments/assets/8e9bf78f-c855-42fa-9dcf-b6d408e1d92f" /></td>
    <td><img width="300" height="250" alt="N4_6_bert_loss" src="https://github.com/user-attachments/assets/e2fa6db6-b502-43e9-b44a-ef0952ece051" /></td>
  </tr>
</table>  
  
From the loss histories, it becomes clear that `N=3` is the optimum value! `N=4` and onwards may lean towards overfitting, while `N=1` and `N=2` may be examples of underfitting - the model hasn't learned any of the underlying trends.  
The testing statistics are as follows...  
| N (Encoder Layers) | Testing Loss | Avg Matches |
|:------------------:|:------------:|:-----------:|
| 1 | 0.940 | 3.681 |
| 2 | 0.940 | 3.534 |
| 3 | 0.891 | 3.849 |
| 4 | 0.872 | 3.833 |  
  
`N=3` performs the best. Note we can account for the low Avg Matches (<3.85/6) simply because the task is below the model's capacity. It achieved a much higher match accuracy when `seq_len=10`.  
  
For reference, here are the attention visualisations for the various N values, for the input `[305, 142, 822, 572, 293, 263]`.  
  
* N=1  (Prediction -> `[3, 0, 4, 4, 2, 2]`)  
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/adea9aa1-9804-4d9d-ba0b-e1803ecd8dce" />  
  
* N=2  (Prediction -> `[2, 1, 4, 4, 2, 2]`)  
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/44aa597e-4167-4b8d-afaf-211eae9330a5" />
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/285a63b1-2e6c-4f07-8f98-2e5887dada60" />  
  
* N=3  (Prediction -> `[3, 0, 4, 4, 2, 1]`)  
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/9073fe93-2528-4c00-9846-fe68b7f792c3" />
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/ae1f0fb8-954c-4528-8e17-b5f0ecdb0825" />
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/63f77e7d-d4cb-42de-a975-777102a407ab" />  
  
* N=4  (Prediction -> `[2, 1, 4, 4, 1, 2]`)  
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/86fb9ae0-b822-4d1c-b493-08c404636823" />
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/4589ebec-cb4e-44a4-9d55-21963c0ecaa7" />
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/d262d607-b20b-4e1e-bcfe-8873b2ad74f9" />
<img width="180" height="180" alt="image" src="https://github.com/user-attachments/assets/d827e562-fd00-427c-b806-adab64ba4332" />  
  
* Expressiveness here can be gauged by how many repititions in the ranks the model is making. `N=3` is not only the most accurate, it is the most expressive of the four!  

   
### Positional Encoding Ablation  
For this ablations, I will be using the "BERT" baseline with `batch_size=512`, `BERTModel(N=3, h=2, dmodel=10, dk=5, dv=5, vocab_size=1000, seq_len=10)`, and an optimizer `AdamW(lr=1e-3,weight_decay=1e-3,betas=(0.9,0.99)))` 
The Positional Encoding call in `models.py` was commented to observe the effects of its absence. Unexpectedly, the model's effectiveness did not diminish. The exact same testing loss and accuracy was observed. The only conclusion that can be drawn from this is that while global position information of the tokens were not provided, the model already learnt to look at the relative positions between pairs and sort them. The task problem seems to have favoured this approach, evident from the pair-wise conclusion from the attention visualisations.

### Varieties in Random Weight Initialisations  
The "BERT" model is inherently very complicated with multiple layers. Even with `N=3`, it has a total of `6 sub-layers`! It is very prone to the problem of exploding/vanishing gradients if the weight initialisations aren't done appropriately. I will be using the "BERT" baseline with `batch_size=512`, `BERTModel(N=3, h=2, dmodel=10, dk=5, dv=5, vocab_size=1000, seq_len=6)`, and an optimizer `AdamW(lr=1e-3,weight_decay=1e-3,betas=(0.9,0.99)))`. My earlier baselines have used Xavier (Uniform) weight initialisations, and I will note how other initialisations compare. A quick note on the different initialisations considered for this-
* Guassian Initialisation `torch.randn()` - Fills the weights with values from the Gaussian normal distribution.
* Xavier (Uniform) Initialisation `torch.utils.init.xavier_uniform_()` - Fill the weights with values using a Xavier uniform distribution.  
&emsp;&emsp; $$a = \text{gain} \times \sqrt{\frac{6}{fan_{in} + fan_{out}}}$$  
* Xavier (Normal) Initialisation `torch.utils.init.xavier_normal_()` - Fill the weights with values using a Xavier normal distribution.  
&emsp;&emsp; $$\text{std} = \text{gain} \times \sqrt{\frac{2}{fan_{in} + fan_{out}}}$$  
* He/Kaiming (Uniform) Initialisation `torch.utils.init.kaiming_uniform_()` - Fill the weights with values using a Kaiming uniform distribution.  
&emsp;&emsp; $$a = \sqrt{\frac{6}{(1 + \text{negative slope}^2) \times fan_{in}}}$$  
* He/Kaiming (Normal) Initialisation `torch.utils.init.kaiming_normal_()` - Fill the weights with values using a Kaiming normal distribution.  
&emsp;&emsp; $$\text{std} = \sqrt{\frac{2}{(1 + \text{negative slope}^2) \times fan_{in}}}$$  
* Orthogonal Initialisation `torch.utils.init.orthogonal_()` - Fill the weights with a (semi) orthogonal matrix.  

The loss history for them are as below...  
<table>
  <tr>
    <td align="center" width="300">Gaussian</td>
    <td align="center" width="300">Uniform Xavier</td>
    <td align="center" width="300">Uniform He/Kaiming</td>
  </tr>
  <tr>
    <td><img width="300" height="250" alt="Gaussian_6_bert_loss" src="https://github.com/user-attachments/assets/6472d2d4-5285-4e61-b4b8-76c963c0dad9" /></td>
    <td><img width="300" height="250" alt="Heuni_6_bert_loss" src="https://github.com/user-attachments/assets/45b006b6-bf70-4e35-bba1-3b457e410ce9" /></td>
    <td><img width="300" height="250" alt="Henorm_6_bert_loss" src="https://github.com/user-attachments/assets/cbd41258-ada9-41aa-8f1b-ea021db0f60f" /></td>
  </tr>
  <tr>
    <td align="center" width="300">Normal Xavier</td>
    <td align="center" width="300">Normal He/Kaiming</td>
    <td align="center" width="300">Orthogonal</td>
  </tr>
  <tr>
    <td><img width="300" height="250" alt="Xaviernorm_6_bert_loss" src="https://github.com/user-attachments/assets/df139e4a-d1a7-4f3f-b600-6bbd3ada3533" /></td>
    <td><img width="300" height="250" alt="Henorm_6_bert_loss" src="https://github.com/user-attachments/assets/aa60687c-1be0-43d5-837a-1ab38b792225" /></td>
    <td><img width="300" height="250" alt="Ortho_6_bert_loss" src="https://github.com/user-attachments/assets/e7f1badf-084c-4095-b2e4-521aebc18563" /></td>
  </tr>
</table>  
Any other kind of initialisation mitigates the exploding gradients that mess with the training as seen in Gaussian initialisation!

## Conclusion

This sub-task explored four architectures for the task of sorted relative rank prediction — MLP, Vanilla RNN, biLSTM, and a Transformer Encoder.  
  
The MLP, while stable and efficient, treats inputs as an unordered collection and fails to capture sequential relationships, leading to repeated rank assignments. The Vanilla RNN improves on this by processing tokens sequentially, but suffers from information loss over longer sequences. The biLSTM addresses this with its gating mechanism and bidirectional processing, achieving the best baseline performance at 6.057/10 average matches.  
  
The Transformer Encoder outperforms all baselines at 8.586/10 average matches, owing to its self-attention mechanism which enables global pairwise comparisons across the entire sequence simultaneously. Attention visualisations confirm that the model has learned meaningful structure — attending to rank-adjacent elements, identifying extremes, etc.  
  
Ablation studies reveal the most proficient of embedding and normalisation approaches. Information about types of weight initialisations, optimizers were observed. Positional encodings in the context of this task seem to not carry much weight. Architecturally, the Transformer scales better with sequence length than any of the baselines, making it the clear choice for tasks requiring global sequence understanding.

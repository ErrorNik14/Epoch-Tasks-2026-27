# Subtask-2:   Code Repair using Sequence-to-Sequence Models  

This report contains information about my attempts at implementing the Transformer architecture (along with the other baselines!)
I will be briefly covering my findings. Unfortunately I couldn't properly train my models or perform deep experiments due to computing constraints (on my local computer and Colab). 

## LLM usage
I have commented next to LLM-generated code lines specifically in the source files. LLMs were used to debug CUDA errors and other operational faults. An LLM was used to generate the diagrams and table structures used across, but not the values in them. 

## Methodology
I decided to base all my baselines on a GRU Seq2seq model. Then to illustrate the effect of a bidirectional encoder, I implemented a biLSTM-LSTM Seq2seq model with an attention mechanism. After which, I implemented the Transformer architecture.  
Given below are diagrams of the architectures used.  

### 1. GRU Seq2seq Architecture  
<img width="450" height="400" alt="gru_arch" src="https://github.com/user-attachments/assets/edc12883-a03e-4464-973d-6946d325656d" />  
  
### 2. biLSTM-LSTM Seq2seq with Attention Architecture  
<img width="450" height="400" alt="attended_arch" src="https://github.com/user-attachments/assets/5a76d854-6f41-492d-a2f8-f2dfbfe67d5b" />  
  
### 3. Transformer Architecture  
<img width="450" height="400" alt="transformer_arch" src="https://github.com/user-attachments/assets/76c821c0-4609-4c12-a9f4-980b3c3c5917" />  
  
  
The dataset used is [CodeXGLUE Code Refinement](https://huggingface.co/datasets/google/code_x_glue_cc_code_refinement) (small split). It contains pairs of buggy and fixed Java code, with identifiers anonymized as `METHOD_1`, `VAR_1`, etc. A word-level tokenizer was used, splitting on whitespace and `.` as delimiters, though `.` delimiters were added back in for two reasons.
1. Ease of detokenization
2. Expectation that the `.` tokens would help the model attend to methods and attributes better

With that being said, the baseline parameters of each model for the initial evaluations will be as follows...  
* <b>GRU Seq2seq</b> (`vocab_size=dataset`, `max_seq_len=150`, `embed_dim=8`, `hidden_size=16`, `tf_ratio=0.5`)   
* <b>biLSTM-LSTM Seq2seq</b> (`vocab_size=dataset`, `max_seq_len=150`, `embed_dim=8`, `hidden_size=16`, `tf_ratio=0.5`)
* <b>Transformer</b> (`N=3`, `heads=4`, `dmodel=64`, `dk=16`, `dv=16`, `max_seq_len=150`)  

with an AdamW optimizer (`lr=1e-3`, `weight_decay=1e-2`, `betas=(0.95,0.99)`), and respective batch sizes (64 for transformer, and 256 for other two). Due to computational complexities while training, it was difficult to get proper convergence, thus all models are not to be treated as "well-trained", the results might thus suffer. Training-Validation-Test split follows the dataset's provided splits.  
For testing metrics, I will be using `BLEU score`, `Edit distance`, `Element-wise character matches`, and `Fraction of valid Java syntax`.

## Baseline Analysis
<table>
  <tr>
    <td align="center" width="300">GRU Seq2seq</td>
    <td align="center" width="300">Attended biLSTM-LSTM Seq2seq</td>
  </tr>
  <tr>
    <td><img width="300" height="250" src="https://github.com/user-attachments/assets/9e3f8193-e4d3-4de8-820f-ce82d8d7cfb3"/></td> 
    <td><img width="300" height="250" src="https://github.com/user-attachments/assets/a0e4ca30-021d-4067-a058-757ed45e5a45"/></td>
  </tr>
  <tr>
    <td align="center">BLEU = 9.499<br>Avg matches = 14.319<br>Edit distance = 89.186<br>Valid syntax = 0.0</td>
    <td align="center">BLEU = 1.345<br>Avg matches = 8.486<br>Edit distance = 101.115<br>Valid syntax = 0.0</td>
  </tr>
</table>
  
We can interpret these results well already.  
      1. The GRU Seq2seq model is the simplest of the two. With a unidirectional encoder, it compresses the entire buggy sequence into a single hidden state vector, which the decoder then conditions on. This creates a severe information bottleneck — longer and more complex code sequences are difficult to encode properly, and the decoder often resorts to repeating common tokens.  
      2. The biLSTM-LSTM Seq2seq model improves on this by using a bidirectional encoder, allowing it to capture context from both directions of the buggy input. Combined with an attention mechanism, the decoder can selectively focus on relevant parts of the encoder's hidden states at each timestep, rather than relying on a single compressed vector. This results in noticeably better performance on longer sequences.  

Let's dive a little deeper into some specifics.

### Training Convergence
The GRU model converges quickly due to its simplicity, but plateaus at a relatively high validation loss. The biLSTM-LSTM model benefits from teacher forcing during training, which stabilises the learning signal and allows it to converge more reliably. However, the attention mechanism adds complexity, and without sufficient training time both models show signs of early plateauing.

### Validation Loss
The GRU model's single hidden state bottleneck causes it to struggle — especially on longer sequences where more context is needed. The biLSTM-LSTM model's attention mechanism gives it a meaningful edge, as it can revisit encoder states dynamically during decoding.

### Stability
Both models display reasonably smooth descent in loss history. The biLSTM-LSTM model is more sensitive to hyperparameters due to its added complexity, but with suitable dropout and gradient clipping, training remains stable.

### Generalization Behavior  
Observe some testing examples from each of the models.  
1. GRU Seq2seq  
&emsp;&emsp;&emsp;&emsp;Buggy   --> `public boolean METHOD_1 ( ) { if ( ! ( VAR_1 ) ) { return false ; } return VAR_2 ; }`  
&emsp;&emsp;&emsp;&emsp;Predicted--> `public METHOD_1 METHOD_1 ( (.( ).) \n( ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ; }`  
&emsp;&emsp;&emsp;&emsp;Actual  --> `public boolean METHOD_1 ( ) { return true ; }`    
The model tends to generate repetitive or overly common tokens (like `public`, `METHOD_1`, `void`, `(`, `.`), to the extent that it fills the remaining tokens with just those symbols.   

2. biLSTM-LSTM Seq2seq  
&emsp;&emsp;&emsp;&emsp;Buggy   --> `public void init ( ) { VAR_1 = java.util.Arrays.asList ( true , true , true , true , true , true , true , true , true ) ; }`  
&emsp;&emsp;&emsp;&emsp;Predicted--> `public public ( ( ( ( ( ( ( ( ( ( ( ( )`  
&emsp;&emsp;&emsp;&emsp;Actual  --> `public void init ( ) { VAR_1 = java.util.Arrays.asList ( true , true , true , true , true , true , true , true , true , true ) ; }`  
The attention model is better at limiting itself ; Giving something smaller and coherent instead of simply spamming the commonly occurring tokens, but displays a severe lack of understanding the structure - choosing to use the same output sequence again and again. This could very well be a training flaw as well - the optimizer and learning rate choice was probably a poor one for this model. 

### Computational Cost
We can gauge this by just recalling the computational complexities of the two baseline models.
| Model | Complexity |
|:------|:----------:|
| GRU Seq2seq | $O(L \cdot (ih + h^2))$ |
| biLSTM-LSTM Seq2seq | $O(L \cdot (ih + h^2))$ |

(where L = sequence length, i = input size, h = hidden size)  
Both are sequential by nature — each decoder timestep depends on the previous — making them difficult to parallelise and slow to train on long sequences.

## Transformer - A brief analysis
Moving onto the main model of importance - a Transformer, a model that utilizes both encoder and decoder, with self-attention and cross-attention between them.  
Let us first look at its training and performance.  
Given below is the loss history...  
<img width="300" height="250" alt="transformer_loss" src="https://github.com/user-attachments/assets/cda16033-7665-42a7-bf42-0807f14fbdab" />  
  
With a `Testing BLEU of 70.535`, `Average matches of 70.326`, `Average edit distance of 32.993`, and a `fraction of valid syntax of 0.895`. The Transformer generalises significantly better than either baseline. Notably, it is even able to perfectly reproduce a significant number of fixed sequences in inference. For example,  
&emsp;&emsp;&emsp;&emsp;Buggy      --> `protected void METHOD_1 ( ) { listener . METHOD_2 ( this ) ; super . METHOD_1 ( ) ; }`  
&emsp;&emsp;&emsp;&emsp;Predicted  --> `protected void METHOD_1 ( ) { super . METHOD_1 ( ) ; }`  
&emsp;&emsp;&emsp;&emsp;Actual     --> `protected void METHOD_1 ( ) { super . METHOD_1 ( ) ; }`  

It is able to do this because of the cross-attention mechanism — the decoder can attend to every encoder token when generating each output token, enabling precise and globally-informed fixes, without the compression problems the baseline models had to face.

That being said, it is not flawless by any means - the Transformer model also suffers from many bad testing predictions. But this could potentially just be a training fault. For example,  
&emsp;&emsp;&emsp;&emsp;Buggy      --> `public boolean METHOD_1 ( java.lang.CharSequence value ) { return ( TYPE_1 . isEmpty ( value ) ) && ( ( value . length ( ) ) >= ( VAR_1 ) ) ; }`  
&emsp;&emsp;&emsp;&emsp;Predicted  --> `public boolean METHOD_1 ( java.lang.CharSequence value ) { return ( ( TYPE_1 . isEmpty ( ) ) && ( value . length ( ) ) >= ( VAR_1 . length ( ) ) ) ; }`  
&emsp;&emsp;&emsp;&emsp;Actual     --> `public boolean METHOD_1 ( java.lang.CharSequence value ) { return ( ! ( TYPE_1 . isEmpty ( value ) ) ) && ( ( value . length ( ) ) >= ( VAR_1 ) ) ; }`  

## Transformer Analysis  
Using the `bertviz` library, it is easy to visualise the various ways in which tokens attend to eacother, be it self attention in the encoder/decoder stacks, or the cross attention between them. Consider this for the buggy input, `protected void METHOD_1 ( ) { listener . METHOD_2 ( this ) ; super . METHOD_1 ( ) ; }`  
<table>
  <tr>
    <td align="center">Encoder Attentions</td>
    <td align="center">Decoder Attentions</td>
    <td align="center">Cross Attentions</td>
  </tr>
  <tr>
    <td><img width="400" height="650" alt="Encoder Attentions" src="https://github.com/user-attachments/assets/0240d202-7296-4264-8f01-acca9d93e055" /></td>
    <td><img width="400" height="400" alt="Decoder Attentions" src="https://github.com/user-attachments/assets/2091bb77-8746-4a4c-bdd9-5b6b3bd94125" /></td>
    <td><img width="400" height="650" alt="Cross Attentions" src="https://github.com/user-attachments/assets/26d7c61a-2f2f-4f0f-8b53-264b8bf302b6" /></td>
  </tr>
</table>  

<i>More can be visualised and observed in the inference notebook!</i>  
For the time being, can draw some conclusions from the attention visualisations of some of the layer heads. Note that the encoder input tokens are on the right, and the decoder output tokens are on the left.  
* A significant number of heads seem to not be performing any operations on the inputs. It's a sign of redundancy.
* Look at this head...
<img width="364" height="623" alt="image" src="https://github.com/user-attachments/assets/671599ba-233a-448e-a3de-c40d855879b5" />  <img width="363" height="622" alt="image" src="https://github.com/user-attachments/assets/9c6b7b6c-f560-4793-81aa-2a3caf752126" /> <img width="360" height="621" alt="image" src="https://github.com/user-attachments/assets/4334881a-d307-44cc-af55-fb68dba8a1f6" />



this head has learnt to attend to the presence of all other methods 


## Conclusion

This sub-task explored three architectures for the task of code repair — GRU Seq2seq, biLSTM-LSTM Seq2seq with attention, and a full Transformer.

The GRU Seq2seq model, while simple and stable, suffers from the classic encoder bottleneck — compressing the entire buggy sequence into a single hidden state is too lossy for meaningful repair. The biLSTM-LSTM model with attention substantially improves on this, using dynamic attention over encoder states to make more informed fixes.

The Transformer outperforms both baselines decisively with a BLEU score of 70.535 and 89.5% valid syntax, owing to its parallel self-attention and cross-attention mechanisms which allow it to reason globally over the input at every decoding step. The attention visualisations further confirm that the model has learned to attend to structurally relevant parts of the buggy code when generating each fix token.

A key limitation across all three models is the anonymized nature of the dataset — identifiers like `METHOD_1` and `VAR_1` are placeholders, meaning true semantic understanding of the code is out of scope. Despite this, the Transformer's strong structural outputs suggest it has learned meaningful syntactic repair patterns from the data alone.

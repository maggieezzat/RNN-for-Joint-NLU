# Tensorflow dynamic seq2seq usage summary (r1.3)

## Motivation:

In fact, almost half a year ago, I wanted to seq2seq of Tensorflow (the blogger went to do something else), the official code has abandoned the version that was implemented with static rnn, and the official website of the tutorial is still based on static rnn. Model, plus the bucket set, [see here] (https://www.tensorflow.org/tutorials/seq2seq)。
![tutorial.png](http://upload-images.jianshu.io/upload_images/1713813-3e90638fd7420d20.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
see it? Is legacy_seq2seq. Originally Tensorflow's implementation of seq2seq is already quite complicated compared to pytorch, and there is no serious tutorial, hehe.
Ok, go back to the topic, solve the problem and solve the problem, find a way to find the best seq2seq solution for Tensorflow**!

## Learning materials
- Well-known blogger WildML wrote a generic seq2seq to google, [document address] (https://google.github.io/seq2seq/), [Github address] (https://github.com/google/seq2seq) . This framework has been adopted by Tensorflow, and our code will be based on the implementation here. But the framework itself is designed to allow users to simply write parameters to build a network, so the document does not have much reference value. We use the code directly to build our own network.
- Russian guy [ematvey] (https://github.com/ematvey) writes: tensorflow-seq2seq-tutorials, [Github address] (https://github.com/ematvey/tensorflow-seq2seq-tutorials). Introducing the use of dynamic rnn to build seq2seq, the decoder uses `raw_rnn`, the principle is similar to the WildML solution. To put it another way, this buddy was also the document of Tustorflow, and wrote such a warehouse as a third-party document, now it is 400+ stars. There are opportunities for loopholes, haha.

## Tensorflow's dynamic rnn
Let's briefly introduce the difference between dynamic rnn and static rnn.
`tf.nn.rnn creates an unrolled graph for a fixed RNN length. That means, if you call tf.nn.rnn with inputs having 200 time steps you are creating a static graph with 200 RNN steps. First, graph creation is slow. Second, you’re unable to pass in longer sequences (> 200) than you’ve originally specified.tf.nn.dynamic_rnn solves this. It uses a tf.While loop to dynamically construct the graph when it is executed. That means graph creation is faster and you can feed batches of variable size.`


From [Whats the difference between tensorflow dynamic_rnn and rnn?] (https://stackoverflow.com/questions/39734146/whats-the-difference-between-tensorflow-dynamic-rnn-and-rnn). That is to say, the static rnn must be expanded in advance, and when executed, the graph is fixed and the maximum length is limited. The dynamic rnn can be cyclically multiplexed at the time of execution.

一In a word, **can use dynamic rnn as much as possible to use dynamic**.

##  Seq2Seq Structure Analysis


![seq2seq.png](http://upload-images.jianshu.io/upload_images/1713813-9260633573ad9e71.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Seq2seq consists of Encoder and Decoder. Generally Encoder and Decoder are based on RNN. Encoder is relatively simple, whether it is multi-layer or bidirectional or replacing a specific Cell, it is easier to use the native API. The difficulty lies in the Decoder: **The input of the rnn cell corresponding to different Decoder is different. For example, in the example above, the input of each cell is the embedding corresponding to the prediction of the cell output at the previous time.

![attention.png](http://upload-images.jianshu.io/upload_images/1713813-ff41a56e2424cbdb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
If Attention is used as shown above, the decoder's cell input also includes the attention weighted summed context.

## Explain by example

![slot filling.png](http://upload-images.jianshu.io/upload_images/1713813-9933e74ae3991048.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
The following is an example of slot filling (a sequence annotation) using seq2seq. Complete code address：https://github.com/applenob/RNN-for-Joint-NLU

## Encoder implementation example
```python
# First construct a single rnn cell
encoder_f_cell = LSTMCell(self.hidden_size)
encoder_b_cell = LSTMCell(self.hidden_size)
 (encoder_fw_outputs, encoder_bw_outputs),
 (encoder_fw_final_state, encoder_bw_final_state) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32, time_major=True)
```
The above code uses `tf.nn.bidirectional_dynamic_rnn` to build a single-layer bidirectional LSTM RNN as the Encoder.
parameter:
- `cell_fw`：Forward lstm cell
- `cell_bw`：Backward lstm cell
- `time_major`：If True, the input needs to be T×B×E, T represents the length of the time series, B represents the batch size, and E represents the dimension of the word vector. Otherwise, it is B × T × E. The output is similar.

return:
- `outputs`：For output on all time series.
- `final_state`：Just the state of the last time node.

一In a word, **The construction of Encoder is to construct an RNN, get the output and the final state.**

## Decoder implementation example
The following highlights how to implement a Decoder using Tensorflow's `tf.contrib.seq2seq`.
In our Decoder here, in addition to the output of the previous time node, each input has the output of the Encoder corresponding to the time node, and the context of the attention.
### Helper
Commonly used `Helper`:
- `TrainingHelper`: A helper for training.
- `InferenceHelper`: A helper for testing.
- `GreedyEmbeddingHelper`: Applicable to the helper using the Greedy strategy sample in the test.
- `CustomHelper`: User-defined helper.

First, let's explain what the helper does: Refer to the Russian brother mentioned above to implement the decoder with `raw_rnn`, and pass a `loop_fn`. This `loop_fn` actually controls each cell at a different time node, gives the output of the last moment, and determines how to input the next moment.
The thing the helper does is basically the same as this `loop_fn`. Here **focuses on** `CustomHelper`, passing in three functions as arguments:
- `initialize_fn`: Returns `finished`, `next_inputs`. Where `finished` is not scala, it is a one-dimensional vector. This function gets the input of the first time node.
- `sample_fn`: Receive parameter `(time, outputs, state)` returns `sample_ids`. That is, according to the output of each cell, how to sample.
- `next_inputs_fn`: Receive parameters `(time, outputs, state, sample_ids)` return `(finished, next_inputs, next_state)`, and determine the input of the next moment based on the output of the previous moment.

## BasicDecoder
With a custom helper, you can define your own Decoder using `tf.contrib.seq2seq.BasicDecoder`. Then use `tf.contrib.seq2seq.dynamic_decode` to execute the decode, and finally return: `(final_outputs, final_state, final_sequence_lengths)`. Where: `final_outputs` is a type of `tf.contrib.seq2seq.BasicDecoderOutput`, including two fields: `rnn_output`, `sample_id`.

## Back to the example

```python
        # Three functions passed to CustomHelper
        def initial_fn():
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            initial_input = tf.concat((sos_step_embedded, encoder_outputs[0]), 1)
            return initial_elements_finished, initial_input

        def sample_fn(time, outputs, state):
            # Select the maximum subscript of logit as the sample
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        def next_inputs_fn(time, outputs, state, sample_ids):
            # The output category on the previous time node, get embedding and then input as the next time node
            pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
            # The input is h_i+o_{i-1}+c_i
            next_input = tf.concat((pred_embedding, encoder_outputs[time]), 1)
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            next_state = state
            return elements_finished, next_inputs, next_state

        # Custom helper
        my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

        def decode(helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                memory = tf.transpose(encoder_outputs, [1, 0, 2])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size, memory=memory,
                    memory_sequence_length=self.encoder_inputs_actual_length)
                cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=self.hidden_size)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attn_cell, self.slot_size, reuse=reuse
                )
                # Decoder using a custom helper
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell, helper=helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=self.batch_size))
                # Get the result of the decode
                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder, output_time_major=True,
                    impute_finished=True, maximum_iterations=self.input_steps
                )
                return final_outputs

        outputs = decode(my_helper, 'decode')
```

## Attntion

The above code, there are still a few places that are not explained：`BahdanauAttention`，`AttentionWrapper`，`OutputProjectionWrapper`。

Start with a simple: `OutputProjectionWrapper` to do a linear mapping, such as the previous cell ouput is T × B × D, D is hidden size, then here to do a linear mapping, directly to T × B × S, where S Is the slot class num. The wrapper internally maintains a variable for linear mapping: `W` and `b`.
![attention.png](http://upload-images.jianshu.io/upload_images/1713813-ff41a56e2424cbdb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

`BahdanauAttention` is an ʻAttentionMechanism`, and the other is: `BahdanauMonotonicAttention`. For the difference between the two, readers should investigate in depth. key parameter:
- `num_units`: Hidden layer dimension.
- `memory`: Usually the output of the RNN encoder
- `memory_sequence_length=None`: The optional parameter, which is the memory mask, exceeds the length data and is not counted in the attention.

Continue to introduce `AttentionWrapper`：This is also a cell wrapper, the key parameters:
- `cell`: Packed cell.
- `attention_mechanism`: The use of the attention mechanism, described above.

![attention.png](http://upload-images.jianshu.io/upload_images/1713813-e9dbf564a7ca6c45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Memory corresponds to the h in the formula, and the output of the wrapper is s.

So what is the specific operation flow of an `AttentionWrapper`? Look at the process given by the official website:

![AttentionWrapper.png](http://upload-images.jianshu.io/upload_images/1713813-28c95c074f1955c8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Loss Function

`tf.contrib.seq2seq.sequence_loss`The loss function of the sequence can be directly calculated. The important parameters are:
- `logits`: Size `[batch_size, sequence_length, num_decoder_symbols]`
- `targets`: Size `[batch_size, sequence_length]`, no need to do one_hot.
- `weights`: `[batch_size, sequence_length]`, ie mask, filter the loss calculation of padding to make the loss calculation more accurate.

## postscript
Only the application of seq2seq in sequence labeling is discussed here. Seq2seq is also widely used in translation and dialog generation, involving generated policy issues such as beam search. I will continue to study later. In addition to the sample strategy, the main techniques of other seq2seq, this article has been basically covered, I hope to help everyone step on the pit.
Complete code: [https://github.com/applenob/RNN-for-Joint-NLU](https://github.com/applenob/RNN-for-Joint-NLU)
# Use TensorFlow to build MLP  
In the [previous work][BP], we built a fully connected multilayer perception (MLP) from scratch. Basically, we only used the `Numpy package` to help with the vectors/matrices calculation. To understand how the back propagation works in process in the MLP is important since the MLP is a vital component in almost all other more "modern" (deep) neural networks (DNN), such as Convolutional Neural Network (CNN) or Recurrent Neural Network (RNN). Even though building a more propose-focused DNN and trying to achieve a higher accuracy by adjusting the hyper-parameters looks a tad like "the Alchemy", observing whether the loss (normally is the "output" of MLP) will still keep reducing after several epochs is still a good way to know the DNN we built "learning" from the data and it will converge to a certain accuracy.  
The "observing-adjusting-observing" method seems to be kinda brute-force, but most time it makes a lot of sense if we can accelerate the calculation process by using some deep learning frameworks, such as [TensorFLow][TF], [PyTorch][Torch], [Keras][Keras], etc. These frameworks can accelerate the tedious matrices calculation by optimizing and distributing them at the backend, and ever taking advantage of the advanced hardware architecture (e.g. GPUs). Here, I used the TensorFlow to reproduce the [previous work][BP], I would like to share some my personal feelings in the process of making my hands dirty as a beginner:
  
  1. TensorFlow is a good choice for deep learning research, at least for me. It's built in a low-level (C++), which means a high flexibility for coding. Actually, the coding logic is as the same as building a computation graph.
  2. Visualization is very very very friendly by using `tensorboard`, by call "tensorboard --logdir=.\log" in the `Terminal`. 
  3. By using the `name_scope` to control the computation graph and `variable_scope` to reuse the shared variables (especially happened in RNNs), the logic flow is very clear.
  4. If I too lazy to invite the wheel for some fundamental function (e.g. MLP here), it also totally fine. The modules embedded in TensorFlow is keep growing rapidly, such as `tf.contrib`, `tf.initializers`, `tf.keras`, etc. 
  5. Easy to manually distribute the computation in different hardwares by using simple lines, e.g. `with tf.device("/gpu:0"):`.
  6. ~~**Something Bad:** I really hate this property. A `tensor` cannot be evaluated unless I put them in certain `session`. It makes sense from the programming language's point of view, but does not make too much sense for me.~~ TensorFlow 大法好， 谷歌真滴强！ 
  





# Convolutional Neural Network  


   [BP]: https://github.com/YIHE1992/Back-Propagation-N.N.-
   [TF]: https://www.tensorflow.org/
   [Torch]: http://pytorch.org/
   [Keras]: https://keras.io/





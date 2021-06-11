+ 运行```bash test_scrip.sh```测试本项目。

  > 输出的表情合成图将位于```./outputs```文件夹中；```./our_output_samples```文件夹下给出了我们的测试输出采样。

+ 运行```bash train_scrip.sh```命令开启新的训练。

  > 训练中checkpoint将保存在```./ckpts```文件夹下，默认每两个epoch保存一次，使用2080 Ti 训练一个epoch的用时为45min。

+ 建议运行时保证空闲显存10G以上。

  > 否则需要在命令中增加选项 ```--ndf [dim_size] -- ngf [dim_size]```， 这里dim_size为模型通道维度参数，默认为64，如果显存不够则需要设置一个更小的值。

+ 建议在Unix环境下运行。

  > 如果在**Windows**中运行需要加入选项参数 ```--serial_batches --n_threads 0```


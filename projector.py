import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
 
from training import misc
 
#----------------------------------------------------------------------------
 
class Projector:
    def __init__(self,
        # 初始化变量
        # vgg16_pkl                       = 'https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2',
        vgg16_pkl=                      '.\models\\vgg16_zhang_perceptual.pkl',
        num_steps                       = 1000,
        initial_learning_rate           = 0.1,
        initial_noise_factor            = 0.05,
        verbose                         = False
    ):
 
        self.vgg16_pkl                  = vgg16_pkl
        self.num_steps                  = num_steps
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = initial_learning_rate
        self.initial_noise_factor       = initial_noise_factor
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = verbose
        self.clone_net                  = True
 
        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None
 
    # 显示信息
    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)
 
    # 设置StyleGAN网络
    def set_network(self, Gs, minibatch_size=1):
        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()
 
        # 向量空间18x512
        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        # 以123为种子，随机产生dlatent_avg_samples=10000个样本值，赋值给latent_samples
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        # 将latent_samples映射到W空间，赋值给dlatent_samples
        # dlatent，即：disentangled latent，映射到18x512向量，可以分别控制不同层次的人脸特征，因此说是disentangled（解纠缠）
        # components.mapping = tflib.Network('G_mapping'......)，见：./training/networks_stylegan2.py
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None) # (10000, 18, 512)
        # 计算均值、方差
        # _dlatent_avg是_dlatent_var优化迭代的初始值
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True) # (1, 18, 512)
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)
 
        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        # 初始化噪声，变量名形如'G_synthesis/noiseXX'，完成后break退出
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            # 将正态分布的随机噪声赋值给v
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            # 计算均值、方差
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            # 正则化
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        # 将噪声组成group
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)
 
        # Image output graph.
        self._info('Building image output graph...')
        # 定义变量_dlatents_var，初始化为0，在start函数中赋初始值为_dlatent_avg
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var') # (1，18，512)
        # 定义输入_noise_in
        self._noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._noise_in  # 与正态分布的随机向量相乘
        # 求和，赋值给_dlatents_expr
        self._dlatents_expr = self._dlatents_var + dlatents_noise
        # components.synthesis将_dlatents_expr生成图像向量
        self._images_expr = self._Gs.components.synthesis.get_output_for(self._dlatents_expr, randomize_noise=False)
 
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # 若生成的图像尺寸超过256x256，把图片调整到256x256
        proc_images_expr = (self._images_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()  # 将proc_images_expr.shape元组转换为list
        if sh[2] > 256:
            factor = sh[2] // 256   # 基于256x256，计算图像的倍数
            # 对超过256x256的像素取平均值，并压缩维度，运算后的shape是[-1, sh[1], sh[2]//factor, sh[2]//factor]
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])
 
        # Loss graph.
        self._info('Building loss graph...')
        # 定义变量_target_images_var
        # 该变量会在下面的start函数中赋初始值
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            # 加载vgg16模型
            self._lpips = misc.load_pkl(self.vgg16_pkl) # vgg16_zhang_perceptual.pkl
        # 使用vgg16模型将proc_images_expr与_target_images_var比较，计算差值
        # 变量_target_images_var会在self.start函数中赋值
        self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
        # 对差值求和，计算_loss
        self._loss = tf.reduce_sum(self._dist)
 
        # Noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                # 沿axis=3滚动一个位置，相乘，平方，沿axis=2滚动一个位置，相乘，取平均数，平方
                # 沿axis=2滚动一个位置，相乘，平方，沿axis=2滚动一个位置，相乘，取平均数，平方
                # 求和
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale，维度减半
                v = tf.reduce_mean(v, axis=[3, 5]) # 求平均数，压缩维度
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight
 
        # Optimizer.
        # 用dnnlib.tflib构建优化函数，定义优化迭代计算_opt_step
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in') # _lrate_in是一个输入项
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in) # 定义优化函数
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars) # 梯度，_loss是输出，[_dlatents_var] + _noise_vars是输入
        self._opt_step = self._opt.apply_updates()
 
    # 运行
    def run(self, target_images):
        # Run to completion.
        # 开始，完成初始化
        self.start(target_images)
        # 迭代
        while self._cur_step < self.num_steps:
            self.step()
 
        # Collect results.
        # 返回结果
        pres = dnnlib.EasyDict()
        pres.dlatents = self.get_dlatents()
        pres.noises = self.get_noises()
        pres.images = self.get_images()
        return pres
 
    # 开始
    def start(self, target_images):
        assert self._Gs is not None
 
        # Prepare target images.
        # 准备目标图像（组），即：优化迭代的目标对象
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        # 如果目标图像尺寸太大，就按照_target_images_var的尺寸缩小到256x256
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
 
        # Initialize optimization state.
        self._info('Initializing optimization state...')
        # 设置_target_images_var变量、_dlatents_var变量
        # 把_dlatent_avg作为_dlatents_var优化迭代的起点，_target_images_var为优化迭代的目标图像（组）
        tflib.set_vars({self._target_images_var: target_images, self._dlatents_var: np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])})
        # 初始化噪声
        tflib.run(self._noise_init_op)
        # 复位优化器状态
        self._opt.reset_optimizer_state()
        # 迭代计数从0开始
        self._cur_step = 0
 
    # 迭代
    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return
        if self._cur_step == 0:
            self._info('Running...')
 
        # Hyperparameters.
        t = self._cur_step / self.num_steps # 完成比例
        # 噪声强度 = 当前dlatent标准差 * 初始噪声因子 * 噪声斜面长度剩余比例的平方
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        # 计算学习率
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp
 
        # Train.
        # 训练，tflib.run
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op) # 每次迭代，随机生成噪声
 
        # Print status.
        # 打印/显示迭代状态
        self._cur_step += 1
        if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
            self._info('%-8d%-12g%-12g' % (self._cur_step, dist_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info('Done.')
 
    def get_cur_step(self):
        return self._cur_step
 
    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0})
 
    def get_noises(self):
        return tflib.run(self._noise_vars)
 
    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0})
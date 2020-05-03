import os
import pickle
import PIL.Image
from PIL import ImageTk
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import glob

# pre-trained network.
Model = './models/stylegan2-ffhq-config-f.pkl'
 
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
_Gs_cache = dict()

# 加载StyleGAN已训练好的网络模型
def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')
 
        _G, _D, Gs = pickle.load(model_file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
 
        # Print network details.
        # Gs.print_layers()
 
        _Gs_cache[model] = Gs
    return _Gs_cache[model]
 
def generate_image(generator, latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))
 
def move_and_show(latent_vector, direction, coeffs):
    global generator,lat_vec,dire
    num=(len(coeffs)+1)//2
    fig,ax = plt.subplots(2, num, figsize=(25, 20), dpi=80)
    lat_vec=latent_vector
    dire=direction
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        if i<num:
            ax[0][i].imshow(generate_image(generator, new_latent_vector))
            ax[0][i].set_title('Coeff: %0.1f' % coeff)
        else:
            ax[1][i-num].imshow(generate_image(generator, new_latent_vector))
            ax[1][i-num].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax[0]]
    [x.axis('off') for x in ax[1]]
    plt.show()

def save_img(favor_coeff):
    global flag,fname,lat_vec
    new_latent_vector = lat_vec.copy()
    new_latent_vector[:8] = (lat_vec + favor_coeff*dire)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    new_person_image = generator.generate_images()[0]
    canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
    temp_img=PIL.Image.fromarray(new_person_image, 'RGB')
    canvas.paste(temp_img, ((0, 0)))
    if flag == 0:
        filename =fname +'_new_age.png'
    if flag == 1:
        filename = fname +'_new_angle.png'
    if flag == 2:
        filename = fname +'_new_gender.png'
    if flag == 3:
        filename = fname +'_new_eyes.png'
    if flag == 4:
        filename = fname +'_new_glasses.png'
    if flag == 5:
        filename = fname +'_new_smile.png'
    if flag==6:
        filename=fname +'_new_white_race.png'
    if flag ==7:
        filename = fname +'_new_yellow_race.png'
    if flag==8:
        filename=fname +'new_black_race.png'

    canvas.save(os.path.join(config.generated_dir, filename))
    npy_name=filename[:-4] + '.npy'
    npy_file=os.path.join(config.src_latents_dir, npy_name)
    np.save(npy_file, new_latent_vector)
    return config.generated_dir+'/'+filename
 
 
def choice(choice,npyfile,filename):
    tflib.init_tf()
    Gs_network = load_Gs(Model)
    global generator,flag, fname
    fname=filename
    flag=choice
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
 
    os.makedirs(config.dlatents_dir, exist_ok=True)
    # person = np.load(os.path.join(config.dlatents_dir, 'Scarlett Johansson01_01.npy'))
    person = np.load(os.path.join(config.src_latents_dir, npyfile))#(1,18,512)
    # Loading already learned latent directions
    direction_list=[]
    direction_list.append(np.load('ffhq_dataset/latent_directions/age.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/angle_horizontal.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/gender.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/eyes_open.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/glasses.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/smile.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/race_white.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/race_yellow.npy'))
    direction_list.append(np.load('ffhq_dataset/latent_directions/race_black.npy'))

    coeffs_list=[]
    coeffs_list.append([-20, -16, -12, -8, 0, 8, 12, 16, 20])
    coeffs_list.append([-40, -32, -24, -16, 0, 16, 24, 32, 40])
    coeffs_list.append([-40, -32, -24, -16, 0, 16, 24, 32, 40])
    coeffs_list.append([-8, -6, -4, -2, 0, 2, 4, 6, 8])
    coeffs_list.append([-16, -12, -8, -4, 0, 4, 8, 12, 16])
    coeffs_list.append([-16, -12, -8, -4, 0, 4, 8, 12, 16])
    coeffs_list.append([-10,-8,-6,-4,-2,0,2,4,6,8,10])
    coeffs_list.append([-10,-8,-6,-4,-2,0,2,4,6,8,10])
    coeffs_list.append([-10,-8,-6,-4,-2,0,2,4,6,8,10])
    move_and_show(person,direction_list[choice],coeffs_list[choice])

def resizeImg(src):
    im = src.resize((256, 256), PIL.Image.ANTIALIAS)  # 调整大小
    imgtk = ImageTk.PhotoImage(image=im)
    return imgtk

def ImageFromVec(npy):
    img_src = np.load(os.path.join(config.src_latents_dir, npy))
    tflib.init_tf()
    Gs_network = load_Gs(Model)
    global generator
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    generator.set_dlatents(img_src)
    print("生成图像...")
    new_image = generator.generate_images()[0]
    temp_img=PIL.Image.fromarray(new_image, 'RGB')
    return resizeImg(temp_img)


def mix_pic(npy1,npy2,psi=0.5,begin=0,end=8):
    os.makedirs(config.generated_dir, exist_ok=True)
    print("载入图像向量1...")
    img_src1= np.load(os.path.join(config.src_latents_dir, npy1))
    print("载入图像向量2...")
    img_src2 = np.load(os.path.join(config.src_latents_dir, npy2))
    tmp_vec=img_src1.copy()
    print("载入混合向量...")
    tmp_vec = img_src1*psi + img_src2*(1-psi)
    new_latent_vector = tmp_vec.reshape((1, 18, 512))
    tflib.init_tf()
    Gs_network = load_Gs(Model)
    global generator
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    generator.set_dlatents(new_latent_vector)
    print("生成图像...")
    new_person_image = generator.generate_images()[0]
    canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
    temp_img=PIL.Image.fromarray(new_person_image, 'RGB')

    filename=npy1[:8]+'_'+npy2[4:8]+'mixed.png'
    print("保存混合图像...")
    canvas.paste(temp_img, ((0, 0)))
    canvas.save(os.path.join(config.generated_dir, filename))
    npy_file=os.path.join(config.src_latents_dir, filename[:-4] + '.npy')
    np.save(npy_file, new_latent_vector)
    print("Done！")
    return resizeImg(temp_img)
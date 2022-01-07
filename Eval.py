import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import net
import time

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, transform_model, content, style, alpha=1,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f, style_f = feat_extractor(vgg, content, style)
    Fccc = transform_model(content_f, content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = transform_model(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc = Fccc[0:1]
    else:
        feat = transform_model(content_f, style_f)
    feat = feat * alpha + Fccc * (1 - alpha)

    return decoder(feat)


def feat_extractor(vgg, content, style):
    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_4.to(device)
    enc_5.to(device)

    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)

    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    content_f = [Content4_1, Content5_1]
    style_f = [Style4_1, Style5_1]

    return content_f, style_f


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder.pth')
parser.add_argument('--transform_path', type=str, default='experiments/mncaf.pth')
parser.add_argument('--style_interpolation_weights', type=str, default="")
parser.add_argument('--a', type=float, default=1.0)
args = parser.parse_args()
args = parser.parse_args()

vgg_path='model/vgg_normalised.pth'


# Additional options
content_size=512
style_size=512
crop='store_true'
# crop=''
save_ext='.jpg'
output_path=args.output

# Advanced options
preserve_color='store_true'
alpha=args.a


preserve_color=False

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --contentDir should be given.

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

if args.content:
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]
print(content_paths)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [args.style]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_paths = [os.path.join(args.style_dir, f) for f in
                   os.listdir(args.style_dir)]

if not os.path.exists(output_path):
    os.mkdir(output_path)

decoder = net.decoder
vgg = net.vgg
network = net.Net(vgg, decoder)
mncaf = network.mncaf

decoder.eval()
mncaf.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder_path))
mncaf.load_state_dict(torch.load(args.transform_path))
vgg.load_state_dict(torch.load(vgg_path))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
mncaf.to(device)
decoder.to(device)

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)
time_x = 0
for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        print("do_inter",do_interpolation)  # 没进来,改了命令进来了了，参考的AdaIN原始推断
        style = torch.stack([style_tf(Image.open(p)) for p in style_paths])
        content = content_tf(Image.open(content_path)) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, mncaf, content, style,
                                    alpha, interpolation_weights)
        output = output.cpu()
        output_name = '{:s}/{:s}_interpolation_{:s}'.format(
            output_path, splitext(basename(content_path))[0], save_ext)
        save_image(output, output_name)

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(content_path))
            style = style_tf(Image.open(style_path))
            if preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                start = time.time()
                output = style_transfer(vgg, decoder, mncaf, content, style,
                                        alpha)
                end = time.time()
                time_x = time_x + end - start
            output = output.cpu()
            print(end - start)
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_path, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
            save_image(output, output_name)
print(time_x)
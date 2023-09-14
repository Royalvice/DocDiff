#!/usr/bin/python
# -*- coding: utf-8 -*-
# This is a watermark tool for images
# Highly inherit from https://github.com/2Dou/watermarker


import argparse
import os
import sys
import math
import textwrap
import random

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops, ImageOps


def randomtext():
    def GBK2312():
        head = random.randint(0xb0, 0xf7)
        body = random.randint(0xa1, 0xfe)
        val = f'{head:x}{body:x}'
        strr = bytes.fromhex(val).decode('gb2312',errors="ignore")
        return strr

    def randomABC():
        A = np.random.randint(65, 91)
        a = np.random.randint(97, 123)
        char = chr(A) + chr(a)
        for i in range(3):
            char = char + str(np.random.randint(0, 9))
        return char

    char = randomABC()
    for i in range(5):
        char = char + GBK2312()
    return char


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def add_mark(imagePath, mark, args):
    """
    添加水印，然后保存图片
    """
    im = Image.open(imagePath)
    im = ImageOps.exif_transpose(im)

    image, mask = mark(im)
    name = os.path.basename(imagePath)
    if image:
        if not os.path.exists(args.out):
            os.mkdir(args.out)

        new_name = os.path.join(args.out, name)
        if os.path.splitext(new_name)[1] != '.png':
            image = image.convert('RGB')
        image.save(new_name, quality=args.quality)
        #mask.save(os.path.join(args.out, os.path.basename(os.path.splitext(new_name)[0]) + '.png'),
         #         quality=args.quality)
        print(name + " Success.")
    else:
        print(name + " Failed.")


def set_opacity(im, opacity):
    """
    设置水印透明度
    """
    assert 0 <= opacity <= 1

    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


def crop_image(im):
    """裁剪图片边缘空白"""
    bg = Image.new(mode='RGBA', size=im.size)
    diff = ImageChops.difference(im, bg)
    del bg
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def gen_mark(args):
    """
    生成mark图片，返回添加水印的函数
    """
    # 字体宽度、高度
    is_height_crop_float = '.' in args.font_height_crop  # not good but work
    width = len(args.mark) * args.size
    if is_height_crop_float:
        height = round(args.size * float(args.font_height_crop))
    else:
        height = int(args.font_height_crop)

    # 创建水印图片(宽度、高度)
    mark = Image.new(mode='RGBA', size=(width, height))

    # 生成文字
    draw_table = ImageDraw.Draw(im=mark)
    draw_table.text(xy=(0, 0),
                    text=args.mark,
                    fill=args.color,
                    font=ImageFont.truetype(args.font_family,
                                            size=args.size))
    del draw_table

    # 裁剪空白
    mark = crop_image(mark)

    # 透明度
    set_opacity(mark, args.opacity)

    def mark_im(im):
        """ 在im图片上添加水印 im为打开的原图"""

        # 计算斜边长度
        c = int(math.sqrt(im.size[0] * im.size[0] + im.size[1] * im.size[1]))

        # 以斜边长度为宽高创建大图（旋转后大图才足以覆盖原图）
        mark2 = Image.new(mode='RGBA', size=(c, c))

        # 在大图上生成水印文字，此处mark为上面生成的水印图片
        y, idx = 0, 0
        while y < c:
            # 制造x坐标错位
            x = -int((mark.size[0] + args.space) * 0.5 * idx)
            idx = (idx + 1) % 2

            while x < c:
                # 在该位置粘贴mark水印图片
                mark2.paste(mark, (x, y))
                x = x + mark.size[0] + args.space
            y = y + mark.size[1] + args.space

        # 将大图旋转一定角度
        mark2 = mark2.rotate(args.angle)

        # 在原图上添加大图水印
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        mask = np.zeros_like(np.array(im))
        mask = Image.fromarray(mask)
        im.paste(mark2,  # 大图
                 (int((im.size[0] - c) / 2), int((im.size[1] - c) / 2)),  # 坐标
                 mask=mark2.split()[3])
        mask.paste(mark2,  # 大图
                   (int((im.size[0] - c) / 2), int((im.size[1] - c) / 2)),  # 坐标
                   mask=mark2.split()[3])
        #mask.show()
        del mark2
        return im, mask

    return mark_im


def main():
    parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument("-f", "--file", type=str, default="./input",
                       help="image file path or directory")
    parse.add_argument("-m", "--mark", type=str, default="测试一句话行不行", help="watermark content")
    parse.add_argument("-o", "--out", default="./2",
                       help="image output directory, default is ./output")
    parse.add_argument("-c", "--color", default="#8B8B1B", type=str,
                       help="text color like '#000000', default is #8B8B1B")
    parse.add_argument("-s", "--space", default=75, type=int,
                       help="space between watermarks, default is 75")
    parse.add_argument("-a", "--angle", default=90, type=int,
                       help="rotate angle of watermarks, default is 30")
    parse.add_argument("--font-family", default="./font/青鸟华光简琥珀.ttf", type=str,
                       help=textwrap.dedent('''\
                       font family of text, default is './font/青鸟华光简琥珀.ttf'
                       using font in system just by font file name
                       for example 'PingFang.ttc', which is default installed on macOS
                       '''))
    parse.add_argument("--font-height-crop", default="1.2", type=str,
                       help=textwrap.dedent('''\
                       change watermark font height crop
                       float will be parsed to factor; int will be parsed to value
                       default is '1.2', meaning 1.2 times font size
                       this useful with CJK font, because line height may be higher than size
                       '''))
    parse.add_argument("--size", default=50, type=int,
                       help="font size of text, default is 50")
    parse.add_argument("--opacity", default=0.15, type=float,
                       help="opacity of watermarks, default is 0.15")
    parse.add_argument("--quality", default=100, type=int,
                       help="quality of output images, default is 90")

    args = parse.parse_args()

    # 随机参数，从[A,B]中随机选取一个整数
    space_ = [30, 40]
    angel_ = [0, 180]
    size_ = [20, 60]
    opacity_ = [50, 95]
    # 字体随机（需要别的字体的话，可以去字体网站下载）
    fonts = [r'./font/青鸟华光简琥珀.ttf', r'./font/楷体_GB2312.ttf', './font/方正仿宋_GBK.TTF', './font/times.ttf',
             './font/simhei.ttf']
    if isinstance(args.mark, str) and sys.version_info[0] < 3:
        args.mark = args.mark.decode("utf-8")

    if os.path.isdir(args.file):
        names = os.listdir(args.file)
        for name in names:
            image_file = os.path.join(args.file, name)
            space = random.randint(space_[0], space_[1])
            args.space = space
            angel = random.randint(angel_[0], angel_[1])
            args.angle = angel
            size = random.randint(size_[0], size_[1])
            args.size = size
            opacity = random.randint(opacity_[0], opacity_[1]) / 100
            args.opacity = opacity
            color = randomcolor()
            args.color = color
            font = np.random.choice(fonts)
            args.font_family = font
            mark_text = randomtext()
            args.mark = mark_text
            mark = gen_mark(args)
            add_mark(image_file, mark, args)
    else:
        mark = gen_mark(args)
        add_mark(args.file, mark, args)


if __name__ == '__main__':
    main()

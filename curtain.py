from collections import namedtuple
from uuid import uuid4
from random import seed, randint
import numpy as np
from random import Random
from PIL import Image
from img_utils import create_multi_gradient_image


Fold = namedtuple('Fold', ('left', 'right', 'center', 'depth'))


class CurtainRandom:
    def __init__(self, curtain_seed: str = None):
        self.seed = curtain_seed or self.gen_seed()
        self.random = Random(self.seed)

    def gen_seed(self):  # noqa
        sd = uuid4()
        sd = sd.hex
        return sd

    def i_0_255(self):
        return self.random.randint(0, 255)

    def f_norm(self, min_, max_):
        return self.random.triangular(min_, max_, ((max_ - min_) / 2) + min_)

    def f_0_1(self):
        return self.random.random()

    def f_0_05(self):
        return self.random.random() * 0.5

    def f_0_025(self):
        return self.random.random() * 0.25

    def f_05_1(self):
        return self.random.random() * 0.5 + 0.5


class Curtain:
    def __init__(self, random: CurtainRandom, size: tuple[int, int] = (1000, 1000)):
        self.rnd = random
        self.size = size
        self.color = self.gen_color()
        self.folds = self.gen_folds()

    def gen_color(self):
        rgb = tuple(self.rnd.i_0_255() for _ in range(3))
        rgb = np.array(rgb, dtype=np.uint8)
        return rgb

    def gen_folds(self):
        pointer = 0
        folds = []

        while True:
            left = pointer + self.rnd.f_0_025()
            center = left + self.rnd.f_0_025()
            right = center + self.rnd.f_0_025()

            if right >= 1.0:
                break

            fold = Fold(
                int(left * self.size[0]),
                int(right * self.size[0]),
                int(center * self.size[0]),
                self.rnd.f_05_1(),
            )
            folds.append(fold)

            pointer = right

        return folds

    def render_folds(self, img):
        for fold in self.folds:
            for x in range(fold.left, fold.right):
                for y in range(self.size[1]):
                    if x <= fold.center:
                        power = ((fold.center - x) / (fold.center - fold.left)) * fold.depth
                    elif x > fold.center:
                        power = ((fold.right - x) / (fold.right - fold.center)) * fold.depth
                    else:
                        raise RuntimeError

                    img[x][y] -= img[x][y] * power

        return img

    def render_color(self, img):
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                img[x][y] = self.color

        return self.render_folds(img)

    def render(self):
        img = np.zeros((self.size[0], self.size[1], 3))
        img = self.render_color(img)
        img = np.minimum(img, 255)
        img = np.maximum(img, 0)
        img = np.array(img, dtype=np.uint8)
        Image.fromarray(img, mode='RGB').save('curtain.png')


if __name__ == '__main__':
    rnd = CurtainRandom()
    curt = Curtain(rnd)
    curt.render()

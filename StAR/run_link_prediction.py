import os

from link_prediction.train import main

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
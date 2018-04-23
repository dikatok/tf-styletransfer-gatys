# A Neural Algorithm of Artistic Style

This is Tensorflow implementation of [A Neural Algorithm of Artistic Style
](https://arxiv.org/abs/1508.06576).

## Directory Structure
```
.
├── core
│   ├── __init__.py
│   ├── loss.py
│   └── model.py
├── optimize.py
├── README.md
├── requirements.txt
```

## Installation
- `pip install -r requirements.txt`
- Download vgg16 weights file from either [here](https://drive.google.com/open?id=1vpyQ855RCRHkO-9oOlo4JLaccS8oguW0) which contains only the required kernels (~55mb) or the full one from [here](http://www.cs.toronto.edu/~frossard/post/vgg16/) which the previous link is based on (~554mb)

## Usage
    Usage:
        python optimize.py [options] <style_image> <content_image>
        
    Options:
        --out_image         Path to save the result
        --learning_rate     Learning rate
        --num_iter          Number of iterations
        --log_iter          Log interval
        --vgg_path          Path to vgg weights
        --content_features  List of features map to be used as content representation
        --style_features    List of features map to be used as style representation
        --content_weight    Content loss weight
        --style_weight      Style loss weight
        
## Example
    python optimize.py style.jpg content.jpg
    
    python optimize.py --style_features conv4_1 conv5_1 --content_features conv4_2 style.jpg content.jpg
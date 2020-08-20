# nannernest


![assets/perfect_sandwich.jpg](assets/perfect_sandwich.jpg)


## Installation

Unfortunately, one of the dependencies `nest2D` requires [boost](https://www.boost.org/) and [cmake](https://cmake.org/), so you have to install these ahead of time. I installed them on my Mac as follows:

```commandline
conda install cmake
brew install boost
```

After that, 

```commandline
pip install nannernest
```


## Usage

Take a top-down picture that contains your banana and at least one slice of bread. Pass the image in via command line:

```commandline
$ nannernest my_image.jpg
```

### CLI Details

```commandline
$ nannernest --help
Usage: nannernest [OPTIONS] IMAGE_PATH

Arguments:
  IMAGE_PATH  Image file which contains bread and banana  [required]

Options:
  --num-slices INTEGER            Maxmimum number of banana slices to consider
                                  [default: 16]

  --banana-pct FLOAT              Percent of banana to cut up  [default: 75]
  --mask-threshold FLOAT          Threshold of segmentation mask.  [default:
                                  0.6]

  --peel-scaler FLOAT             Fraction of slice that is assumed to belong
                                  to banana insides versus the peel.
                                  [default: 0.8]

  --ellipse-ratio FLOAT           Assumed ratio of minor axis to major axis of
                                  banana slice ellipses  [default: 0.85]

  --plot-segmentation / --no-plot-segmentation
                                  Whether or not to plot the segmentation
                                  masks  [default: False]

  --plot-slicing / --no-plot-slicing
                                  Whether or not to plot the slicing circle
                                  and skeleton  [default: False]

  --output TEXT                   Name of file to output  [default:
                                  perfect_sandwich.jpg]
```

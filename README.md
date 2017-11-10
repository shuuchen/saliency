# saliency
A python command line tool to create Itty-Koch-Style saliency maps.

## Requirements
* Python 2 or 3
* Opencv 2+

## Command options

* Find all the options by -h or --help
```python
python saliency.py -h
```

## How to use
* For the input image
<p>
  <img src="https://github.com/shuuchen/saliency/blob/master/data/quad_video0_0.jpg" height="216" width="396" />
</p>

* Simply specify the input/output directories
```python
python saliency.py --inputDir [input_dir] --outputDir [output_dir]
```
<p>
  <img src="https://github.com/shuuchen/saliency/blob/master/data/quad_video0_0_saliency.jpg" height="216" width="396" />
</p>

* You can also mark the maximum saliency in the output image
```python
python saliency.py --inputDir [input_dir] --outputDir [output_dir] --markMaxima
```
* Some recommendations for parameter tuning
  * Image size

<p>
  <img src="https://github.com/shuuchen/saliency/blob/master/data/quad_video0_0_saliency_maxima.jpg" height="216" width="396" />
</p>

## Papers
* A Model of Saliency-Based Visual Attention for Rapid Scene Analysis , Laurent Itti, Christof Koch, and Ernst Niebur, PAMI, 1998

## Thanks
* https://gist.github.com/tatome/d491c8b1ec5ed8d4744c

## License
* Released under the MIT license. See LICENSE for details.

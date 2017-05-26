### Install Dependencies
- Install python dependencies
```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
cd experiments
pip install --user -r requirements.txt 
```
- Install the deepmind/torch-hdf5 which gives HDF5 bindings for Torch:
```bash
luarocks install https://raw.githubusercontent.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec
```
### Download and Prepare the Data
- Download the dataset
```bash
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip train2014.zip
unzip val2014.zip
```
- Create HDF5 file
```bash
python make_style_dataset.py \
  --train_dir train2014 \
  --val_dir val2014 \
  --output_file data.h5
  ```
- Download the VGG-16 Torch model
```bash
bash models/download_models.sh 
```

### Train the Model
- Training a 16 style model. For customized styles, set ``style_image_folder`` pointing at the folders containing the style images. 
```bash
th main.lua \
-h5_file data.h5 \
-style_image_folder images/9styles \
-style_image_size 512 \
-checkpoint_name 9styles \
-gpu 0
```
- Test the model
```bash
th test.lua \
-premodel 9styles.t7 \
-input_image images/content/venice-boat.jpg
```

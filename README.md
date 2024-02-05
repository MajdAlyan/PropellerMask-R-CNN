Bilder werden mit Vgg Image Annotator(VIA) annotiert. Das Bedeutet, in diesem Schritt werden Segemente Von den Bildern klassifiziert als 
(Tip vortex cavitation, Hub vortex cavitation, Cloud cavitation, Blade root cavitation, Bubble cavitation, Sheet cavitation, Propeller hull vortex cavitation )
Das ist hier eine Anleitung, wie die Umgebung von Mask_RCNN unter Windows with Anaconda läuft. man muss die richtige versionen von den 
Bibliotheken installieren, weil man oft fehler bekommt. 
mkdir MaskRCNN
cd MaskRCNN
conda create --name Matterprot_MaskRCNN python=3.6.13 tensorflow==1.15.0 Keras==2.2.4 h5py==2.8.0 pip
conda activate Matterprot_MaskRCNN
conda install -c conda-forge jupyterlab
git clone https://github.com/matterport/Mask_RCNN.git
cd  Mask_RCNN
pip install -r requirements.txt
git clone https://github.com/philferriere/cocoapi.git
pip install pycocotools
Damit die Bibliotheken richtig von Coco(cocoapi, pycocotools) funktionieren, 
muss man sicherstellen, dass Visual Studio 2022 oder 2019 oder 2015 installiert ist.
der Fehler ist: error: command ´cl.exe´ failed: No such file or directory 
Link: https://stackoverflow.com/questions/41724445/python-pip-on-windows-command-cl-exe-failed
hier wird erklärt, wieso man cl.exe installieren muss 
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5 > Mask_RCNN/mask_rcnn_balloon.h5. Das Funktioniert nicht  
python setup.py install
jupyter lab
open samples/demo.ipynb 


surpress warnings:
import warnings
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)

final:
conda deactivate

tensorboard --logdir="C:\Users\majd4\Desktop\Bachelorarbeit\Bachelor-Arbeit-Daten\MaskRCNNProjekt\MaskRCNN_2\Mask_RCNN\logs"

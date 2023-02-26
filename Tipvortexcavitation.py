"""
Mask R-CNN wird mit Tipvortexcavitationdatensatz trainiert. Farbspritzereffekt(color splash effect) wird im Algorithmus Implementiert

------------------------------------------------------------

# Ein neues Modell wird mit den vortrainierten Coco-Gewichten trainiert
für Linux der Pfad und die Gewichte von Coco dataset
python3 Tipvortexcavitation.py train --dataset=/path/to/Tipvortexcavitation/dataset --weights=coco 
Windows
python  Tipvortexcavitation.py train --dataset=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/datasets/Tipvortexcavitation --weights=coco

# Fortsetzen des Trainierens von einem Modell, was schon trainiert wurde 
python3 Tipvortexcavitation.py train --dataset=/path/to/Tipvortexcavitation/dataset --weights=last für Linux 
python  Tipvortexcavitation.py train --dataset=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/datasets/Tipvortexcavitation --weights=last

# Ein neues Modell wird mit den ImageNet-Gewichten trainiert
python3 Tipvortexcavitation.py train --dataset=/path/to/Tipvortexcavitation/dataset --weights=imagenet für Linux
python  Tipvortexcavitation.py train --dataset=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/datasets/Tipvortexcavitation --weights=imagenet
    
# Farbspritzer(color splash) auf ein Bild anwenden
python3 Tipvortexcavitation.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file> für Linux
python Tipvortexcavitation.py  splash --weights=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/tipvortexcavitation20220425T1108.h5 --image=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/88.jpg
    
# Farbspritzer (color splash) anwenden mit den zuletzt trainierten Gewichten
python3 Tipvortexcavitation.py splash --weights=last --video=<URL or path to file> für Linux
python  Tipvortexcavitation.py splash --weights=last --video=<URL or path to file>  für Windows
python Tipvortexcavitation.py splash --weights=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/mask_rcnn_tipvortexcavitation_0100.h5 --video=C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN/samples/Tipvortexcavitation/33488.mp4
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw


# der Pfad der Datei von dem Mask R-CNN Projekt 
ROOT_DIR = os.path.abspath("C:/Users/majd4/Desktop/Bachelorarbeit/Bachelor-Arbeit-Daten/MaskRCNNProjekt/MaskRCNN_2/Mask_RCNN")

# Maske RCNN importieren
sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Pfad zur Datei mit den trainierten Gewichten
# was vorher trainiert wurde, natürlich wird es nochmals benutzt, um ein neues Modell zu trainieren 
# es wird das neue Modell besser in der erkennung von Tipvortexcavitation
WEIGHTSS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# hier ist die Datei, wo die logs gespeichert wurden 
# Epochen von dem Training 
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Konfigurationen
############################################################

# Esi ist hilfreich diese Website zu lesen, wenn man Änderung auf den Code vornehmen möchte 
#  https://github.com/matterport/Mask_RCNN/wiki


class TipvortexcavitationConfig(Config):
    """Konfiguration für das trainieren mit dem Tipvortexcavitationdatensatz
    Abgeleitet von dem basic Config Klasse und werden einige Werte überschrieben 
    """
    # ANZAHL DER zu verwendenden GPUs. Wenn nur eine CPU verwendet wird, muss dies auf 1 gesetzt werden.
    # es ist abhängig von dem Prozessor und die Grafikkarte, die man hat 
    # hier Auf meinem Laptop nutze ich AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx 2.30 GHz
    GPU_COUNT = 1
    
    #Die Konfiguration wird einen erkennbaren Namen gegeben 
    NAME = "Tipvortexcavitation"

    # verwende eine GPU mit 12 GB Speicher, die zwei Bilder aufnehmen kann.
    # Passe nach unten an, wenn eine kleinere GPU verwendet wird.
    IMAGES_PER_GPU = 1

    # Anzahl der Klassen (einschließlich Hintergrund)
    NUM_CLASSES = 1 + 1  # Background + Tipvortexcavitation
    
    # Anzahl der Farbkanäle pro Bild.  
    # RGB = 3 (Bilder nur mit Farbe), 
    # grayscale = 1 (Bilder nur Schwarz und Weiß), 
    # RGB-D = 4 ( Bilder werden von einer Speziellen Kamera aufgenommen. 
    # jedes Pixel im Bildem ist mit einem Intensitätswert verbunden)
    IMAGE_CHANNEL_COUNT = 3
    
    
    # Backbone Netzwerk die Architektur
    # Unterstützte Werte sind: resnet50, resnet101.
    BACKBONE = "resnet101"

    # Anzahl der Trainingsschritte pro Epoche
    STEPS_PER_EPOCH = 10
    # Die learnrate
    LEARNING_RATE = 0.001
    
    # Anzahl der Validierungsschritte, die am Ende jeder Trainingsepoche ausgeführt werden.
    
    VALIDATION_STEPS = 20
    
"""
class Config(object):
    Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0
    
""" 
    
    
############################################################
#  Datensatz
############################################################

class TipvortexcavitationDataset(utils.Dataset):

    def load_Tipvortexcavitation(self, dataset_dir, subset):
        """Lade eine Teilmenge des Tip Vortex Cavitation-Datensatzes.
        """
        # Klassen hinzufügen.  
        self.add_class("Tipvortexcavitation", 1, "Tipvortexcavitation")

        # Trainings- oder Validierungsdatensatz?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

       
        annotations = json.load(open(os.path.join(dataset_dir, "Tipvortexcavitation_json.json")))
        annotations = list(annotations.values())  # 

        
        # Annotationen.unannotierte Bilder überspringen.
        annotations = [a for a in annotations if a['regions']]

        # Bilder hinzufügen
        for a in annotations:
            # Erhalte die x, y Koordinaten der Punkte der Polygone
            # Die Polygone werden in the shape_attributes gespeichert
            if type(a['regions']) is dict: #annotated
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else: # unanotated
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() benötigt die Bildgröße, um Polygone in Masken zu konvertieren.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Tipvortexcavitation",
                image_id=a['filename'],  
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Erzeuge Instanz  für ein Bild.
       zurückgegeben:
        
        """
        
        image_info = self.image_info[image_id]
        if image_info["source"] != "Tipvortexcavitation":
            return super(self.__class__, self).load_mask(image_id)

        # Erzeuge Polygone zum einem bitmap mask of shape
        # [Höhe, Breite, Instanz_Anzahl]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        
       
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Gibt den Pfad des Bildes zurück."""
        info = self.image_info[image_id]
        if info["source"] == "Tipvortexcavitation":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

   
def train(model):
    """ das Modell trainieren."""
    # Trainingsdatensatz(Trainingsdatenmenge).
    dataset_train = TipvortexcavitationDataset()
    dataset_train.load_Tipvortexcavitation(args.dataset, "train")
    dataset_train.prepare()

    # Validierungsdatensatz(Validierungsdatenmenge)
    dataset_val = TipvortexcavitationDataset()
    dataset_val.load_Tipvortexcavitation(args.dataset, "val")
    dataset_val.prepare()

    
    
    
    
    # keine Notwendigkeit, alle Schichten zu trainieren. 
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
    
    """
    # man muss auswählen, welche layers man trainieren möchte
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
    """


def color_splash(image, mask):
    """Farbspritzer-Effekt anwenden.
    Bild: RGB Bild [Höhe, Breite, 3]
    mask:Instanzsegmentierung mask [Höhe, Breite, Anzahl der Instanzen]

    es wird schwarz weißes Bild mit Farbe zurückgegeben 
    """
    
    # Erstelle ein schwarzweiße Kopie von dem Farbigen Bild.   
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Kopiere die gefärbte Maske von dem Originalbild  
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Bild oder Video?
    if image_path:
        
        # das Modell für erkennung ausführen und erzeuge Farbspritzereffekt(color splash effect)
        print("Running on {}".format(args.image))
        # Bild lesen
        image = skimage.io.imread(args.image)
        #  Objekte erkennen
        r = model.detect([image], verbose=1)[0]
        # Farbspritzer
        splash = color_splash(image, r['masks'])
        # Ausgabe speichern 
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Das nächste Bild lesen
            success, image = vcapture.read()
            if success:
                # OpenCV gibt die Bilder als BGR zurück und wandelt zu RGB um 
                image = image[..., ::-1]
                # Objekte erkennen
                r = model.detect([image], verbose=0)[0]
                # Farbspritzer(Farbe spritzen)
                splash = color_splash(image, r['masks'])
                # RGB -> BGR um ein Bild in einem Video zu speichern
                splash = splash[..., ::-1]
                # dem Videoschreiber ein Bild hinzufügen
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Speichere zu ", file_name)


############################################################
#  Trainieren
############################################################

if __name__ == '__main__':
    import argparse

    # 
    parser = argparse.ArgumentParser(
        description='Trainiere Mask R-CNN, um Tipvortexcavitation zu erkennen')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Tipvortexcavitation dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Argumente validieren
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Konfigurationen
    if args.command == "train":
        config = TipvortexcavitationConfig()
    else:
        class InferenceConfig(TipvortexcavitationConfig):
            # setz die Batchgröße(batch size) auf 1.da wir führen Inferenz auf ein Bild nach dem anderen aus 
            # Batchgröße(Batch size). Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # ein Modell erzeugen 
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Gewichte Datei zum Laden auswählen 
    if args.weights.lower() == "coco":
        weights_path = WEIGHTSS_PATH
        # Gewichte Datei herunterladen
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # letzte trainierte Gewichte finden oder das letzte trainierte Modell mit den Gewichten finden
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Das neue Modell wird mit den Gewichten von Imagenetmodell Initialisieren
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Gewichte laden
    print("Gewichte laden ", weights_path)
    if args.weights.lower() == "coco":
        
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Trainieren or evaluieren
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' ist nicht erkannt. "
              "benutze 'trainieren' oder 'splash'".format(args.command))

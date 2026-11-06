import torch, torch.nn as nn

CLASS_NAMES = [
 "red and white circle 20 kph speed limit","red and white circle 30 kph speed limit",
 "red and white circle 50 kph speed limit","red and white circle 60 kph speed limit",
 "red and white circle 70 kph speed limit","red and white circle 80 kph speed limit",
 "end / de-restriction of 80 kph speed limit","red and white circle 100 kph speed limit",
 "red and white circle 120 kph speed limit","red and white circle red car and black car no passing",
 "red and white circle red truck and black car no passing","red and white triangle road intersection warning",
 "white and yellow diamond priority road","red and white upside down triangle yield right-of-way",
 "stop","empty red and white circle","red and white circle no truck entry",
 "red circle with white horizonal stripe no entry","red and white triangle with exclamation mark warning",
 "red and white triangle with black left curve approaching warning","red and white triangle with black right curve approaching warning",
 "red and white triangle with black double curve approaching warning","red and white triangle rough / bumpy road warning",
 "red and white triangle car skidding / slipping warning","red and white triangle with merging / narrow lanes warning",
 "red and white triangle with person digging / construction / road work","red and white triangle with traffic light approaching warning",
 "red and white triangle with person walking warning","red and white triangle with child and person walking warning",
 "red and white triangle with bicyle warning","red and white triangle with snowflake / ice warning",
 "red and white triangle with deer warning","white circle with gray strike bar no speed limit",
 "blue circle with white right turn arrow mandatory","blue circle with white left turn arrow mandatory",
 "blue circle with white forward arrow mandatory","blue circle with white forward or right arrow mandatory",
 "blue circle with white forward or left arrow mandatory","blue circle with white keep right arrow mandatory",
 "blue circle with white keep left arrow mandatory","blue circle with white roundabout arrows mandatory",
 "end of no passing zone","end of no passing by trucks zone",
]

class CompactCNN(nn.Module):
    def __init__(self, n_classes=43, p_drop=0.30):
        super().__init__()
        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(
            *blk(3, 32), *blk(32, 32), nn.MaxPool2d(2),
            *blk(32, 64), *blk(64, 64), nn.MaxPool2d(2),
            *blk(64, 128), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(p_drop), nn.Linear(256, n_classes))
    def forward(self, x):
        return self.classifier(self.features(x))

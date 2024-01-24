from torchsummary import summary
# from torch_model import no_def_model
#from torch_practice import no_def_model
#from torch_model import no_def_model
from torch_model_modified import no_def_model
model = no_def_model(20)
summary(model, (1, 5000))

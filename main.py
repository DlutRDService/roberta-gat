import torch
from model import RobertaGAT


if __name__ == "__main__":
    model = RobertaGAT(roberta_model_name="roberta-base", num_classes=4)
    model.load_state_dict(torch.load('./model/model.pth'))

    model.eval()



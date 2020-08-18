import torch
import torch.nn 
import torch.nn.functional as F

class ValueMatrixNetwork(nn.Module):
    def __init__(self):
        super(ValueMatrixNetwork, self).__init__()
        self.conv1

    def forward(self, belief, instruct):
        x1 = self.map_embeds(belief)
        x2 = self.lang_embeds(instruct)
        attn = F.softmax(self.attend(x2))
        
        latent = torch.bmm(attn, x1)

        # Make decoder have enough layers for path length
        y = self.decoder(latent)
        return y

    def map_embeds(self, belief):
        pass

    def lang_embeds(self, instruct):
        pass 

    def attend(self, lang):
        """
        input of lang dim
        return of image dim
        """
        pass

    def decoder(self, latent):
        pass 

    




# from data import train_data, test_data
if __name__ == "__main__":
    model = ValueMatrixNetwork()
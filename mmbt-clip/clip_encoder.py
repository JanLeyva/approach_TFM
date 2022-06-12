class ClipEncoderMulti(nn.Module):
    def __init__(self, num_embeds, num_features=image_features_size):
        super().__init__()        
        self.model = clip_model
        self.num_embeds = num_embeds
        self.num_features = num_features

    def forward(self, x):
        # 4x3x288x288 -> 1x4x640
        out = self.model.encode_image(x.view(-1,3,288,288))
        out = out.view(-1, self.num_embeds, self.num_features).float()
        return out  # Bx4x640
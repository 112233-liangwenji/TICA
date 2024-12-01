def forward(self):
    self.pred = self.encoder(self.image, self.flow)
   
def new_loss():
    with torch.no_grad():
        self.pred_aug = self.encoder(self.image_aug)
    

    mask = torch.sigmoid(self.pred)
    mask_aug = torch.sigmoid(self.pred_aug)

    inter = torch.logical_and(mask > 0.5, mask_previous > 0.5)
    fg_m = mask[inter]
    fg_mp = mask_aug[inter]

    # kl divergence
    p_fg, p_bg = torch.sigmoid(pred[inter]).unsqueeze(0), 1 - torch.sigmoid(pred[inter]).unsqueeze(0)
    pp_fg, pp_bg = torch.sigmoid(pred_previous[inter]).unsqueeze(0), 1 - torch.sigmoid(pred_previous[inter]).unsqueeze(0)
    p, pp = torch.cat([p_fg, p_bg], dim=0), torch.cat([pp_fg, pp_bg], dim=0)
    loss = F.kl_div(torch.log(p), pp, reduction='mean')         
       
    # l1 loss
    loss = F.l1_loss(mask, mask_aug, reduction='mean')

import torch
import torch.nn as nn
import torch.nn.functional as F


class WSSSNetwork(nn.Module):
    def __init__(self, encoder, decoder, projector, num_classes=20):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector

        self.max_pool2d = nn.AdaptiveMaxPool2d((1, 1))
        _dim = self.encoder.embed_dim
        self.top_classifier = nn.Conv2d(_dim, num_classes - 1, kernel_size=1, bias=False)
        self.aux_classifier = nn.Conv2d(_dim, num_classes - 1, kernel_size=1, bias=False)

    def get_param_groups(self):
        param_groups = [[], [], [], []]

        skip_list = self.encoder.no_weight_decay()
        for name, param in self.encoder.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                param_groups[0].append(param)  # no weight decay
            else:
                param_groups[1].append(param)

        for name, param in self.decoder.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias"):
                param_groups[2].append(param)  # no weight decay
            else:
                param_groups[3].append(param)

        for name, param in self.projector.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias"):
                param_groups[2].append(param)  # no weight decay
            else:
                param_groups[3].append(param)

        param_groups[3].append(self.top_classifier.weight)
        param_groups[3].append(self.aux_classifier.weight)

        return param_groups

    def to_2D(self, x, h, w):
        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], h, w)
        return x

    def forward(self, x, mask=None, cam_only=False):
        multi_views = True if isinstance(x, (tuple, list)) else False
        if multi_views:
            x1 = torch.cat(x[:2], dim=0)
            x2 = torch.cat(x[2:], dim=0) if len(x[2:]) > 0 else None
        else:
            x1 = x
            x2 = None
        H, W = x1.shape[-2:]

        # encoder + projector
        top_enc_out, aux_enc_out = self.encoder(x1, mask=mask)
        project_out = self.projector(top_enc_out)

        if x2 is not None:
            top_enc_out2 = self.encoder(x2)[0]
            project_out2 = self.projector(top_enc_out2[:, 0])
        else:
            project_out2 = None

        top_enc_out = top_enc_out[:, 1:]  # remove [CLS] token
        if multi_views:  # first view only
            top_enc_out = top_enc_out.chunk(2)[0]
            aux_enc_out = aux_enc_out.chunk(2)[0]
        ps = self.encoder.patch_size
        top_enc_out = self.to_2D(top_enc_out, H // ps, W // ps)
        aux_enc_out = self.to_2D(aux_enc_out, H // ps, W // ps)
        top_fmap = top_enc_out
        aux_fmap = aux_enc_out

        with torch.no_grad():
            top_cam = F.relu(F.conv2d(top_fmap, self.top_classifier.weight))
            aux_cam = F.relu(F.conv2d(aux_fmap, self.aux_classifier.weight))
        if cam_only:
            return top_cam, aux_cam

        # segmentation
        seg_out = self.decoder(top_enc_out)

        # classification
        top_cls_out = self.max_pool2d(top_enc_out)
        top_cls_out = self.top_classifier(top_cls_out).flatten(1)

        aux_cls_out = self.max_pool2d(aux_enc_out)
        aux_cls_out = self.aux_classifier(aux_cls_out).flatten(1)

        return {
            "top_cls_out": top_cls_out,
            "aux_cls_out": aux_cls_out,
            "project_out": project_out,
            "project_out2": project_out2,
            "seg_out": seg_out,
            "top_cam": top_cam,
            "aux_cam": aux_cam,
            "top_fmap": top_fmap,
            "aux_fmap": aux_fmap,
        }


def build_model(args, pretrained=False):
    from .decoder import LargeFOV
    from .projector import DINOHead
    from .vit import vit_base_patch16_224

    encoder = vit_base_patch16_224(
        pretrained=pretrained,
        img_size=args.input_size,
        cls_depth=args.cls_depth,
        drop_path_rate=args.drop_path_rate,
    )
    decoder = LargeFOV(encoder.embed_dim, args.num_classes)
    projector = DINOHead(encoder.embed_dim, args.out_dim)

    model = WSSSNetwork(encoder=encoder, decoder=decoder, projector=projector, num_classes=args.num_classes)
    return model

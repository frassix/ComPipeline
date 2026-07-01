import torch
import torch.nn.functional as F

DEFAULT_DROPOUT = 0.4
DEFAULT_MAT_DIFF_SCALE = 0.001
IDENTITY_3X3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]

def _layer_bn_relu_mask(x, nn, bn, mask=None):
    x = F.relu(bn(nn(x)))

    if mask is not None:
        x = x * mask[:, None, :]

    return x

def generate_mask(x):
    B, D, N = x.shape

    # if D != 4:
        # raise ValueError("Position + Feature Dimension is not 4")
    
    mask = torch.zeros([B, N], dtype=torch.int, device=x.device)
    npts_all = torch.zeros(B, dtype=torch.int, device=x.device)

    # Count number of non-zero energies
    for i in range(B):
        npts = torch.count_nonzero(x[i, 3, :])
        mask[i, :npts] = 1
        npts_all[i] = npts

    return mask, npts_all

def generate_mask_simple(x):
    mask = (x[:, -1, :] != 0).int()
    npts_all = torch.sum(x, 1)

    return mask, npts_all

class STN3d(torch.nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 9)
        
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.register_buffer("identify_3x3", torch.eye(3).view(1, 9))

    def forward(self, x, mask=None):
        B, D, N = x.size()
        
        if mask is None:
            mask = torch.ones([B, N], dtype=torch.int, device=x.device)

        x = _layer_bn_relu_mask(x, self.conv1, self.bn1, mask=mask)
        x = _layer_bn_relu_mask(x, self.conv2, self.bn2, mask=mask)
        x = _layer_bn_relu_mask(x, self.conv3, self.bn3, mask=mask)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = _layer_bn_relu_mask(x, self.fc1, self.bn4)
        x = _layer_bn_relu_mask(x, self.fc2, self.bn5)
        x = self.fc3(x)

        # iden = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32, device=x.device).view(1, 9).repeat(B, 1)
        # iden = torch.eye(3, dtype=torch.float32, device=x.device).flatten().view(1, 9).repeat(B, 1)
        iden = self.identity_3x3.repeat(B, 1)

        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(torch.nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, k * k)
        
        self.relu = torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)

        self.k = k

        self.register_buffer("identity_kxk",
                             torch.eye(k).view(1, k*k))

    def forward(self, x, mask=None):
        B, D, N = x.size()
        
        if mask is None:
            mask = torch.ones([B, N], dtype=torch.int, device=x.device)

        x = _layer_bn_relu_mask(x, self.conv1, self.bn1, mask=mask)

        x = _layer_bn_relu_mask(x, self.conv2, self.bn2, mask=mask)
        
        x = _layer_bn_relu_mask(x, self.conv3, self.bn3, mask=mask)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = _layer_bn_relu_mask(x, self.fc1, self.bn4)
        x = _layer_bn_relu_mask(x, self.fc2, self.bn5)

        x = self.fc3(x)

        iden = self.identity_kxk.repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(torch.nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super().__init__()

        # Input T-Net active
        self.stn = STNkd(3)

        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv1b = torch.nn.Conv1d(64, 64, 1)
        self.conv2a = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn1b = torch.nn.BatchNorm1d(64)
        self.bn2a = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.global_feat = global_feat

        # Feature T-Net NOT active
        self.feature_transform = False
        self.fstn = None

    def forward(self, x, mask=None):
        B, D, N = x.size()

        if mask is None:
            mask = torch.ones([B, N], dtype=torch.int, device=x.device)
        
	# ===== Input T-Net: 3x3 matrix on (x,y,z) =====
        # x shape: [B, 4, N], take the first 3 channels
        trans = self.stn(x[:, :3, :], mask=mask)  # [B, 3, 3]

        # To [B, N, 4]
        x_t = x.transpose(2, 1)

        # Dividing positions and feature
        pos = x_t[:, :, :3]    # [B, N, 3]
        feat = x_t[:, :, 3:]   # [B, N, 1] (energy)

        # apply trasformation to positions
        pos = torch.bmm(pos, trans)  # [B, N, 3]

        # from [B, N, 4] to [B, 4, N]
        x_t = torch.cat([pos, feat], dim=2)
        x = x_t.transpose(2, 1)

        # ===== PointNet without Feature T-Net =====
        x = _layer_bn_relu_mask(x, self.conv1, self.bn1, mask=mask)
        x = _layer_bn_relu_mask(x, self.conv1b, self.bn1b, mask=mask)

        # Feature T-net NOT used
        trans_feat = None
        pointfeat = x

        x = _layer_bn_relu_mask(x, self.conv2a, self.bn2a, mask=mask)
        x = _layer_bn_relu_mask(x, self.conv2, self.bn2, mask=mask)
        x = _layer_bn_relu_mask(x, self.conv3, self.bn3, mask=mask)

       
        x = torch.max(x, 2)[0] # global max pooling 

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet(torch.nn.Module):
    def __init__(self, nclass=1, add_nhits=False):
        super().__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=4)

        nlin = 1025 if add_nhits else 1024
        self.fc1 = torch.nn.Linear(nlin, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, nclass)

        self.dropout = torch.nn.Dropout(DEFAULT_DROPOUT)

        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        
        self.relu = torch.nn.ReLU()

        self.add_nhits = add_nhits

    def forward(self, x, mask=None):
        B, D, N = x.shape

        if D != 4:
            raise ValueError(f"Expected 4 channels (x, y, z, energy), got {D}")
        
        if mask is not None and mask.shape != (B, N):
            raise ValueError("Wrong shape for input mask")

        if mask is None:
            # mask, npts = generate_mask(x)
            mask = torch.ones([B, N], dtype=torch.int, device=x.device)

        # Get the number of hits for each entry in the batch
        energy = x[:, 3, :]
        nhits = torch.count_nonzero(energy, dim=1)

        x, trans, trans_feat = self.feat(x, mask=mask)

        # Appending the nhits value to the input to the linear layers
        if self.add_nhits:
            nhits_expanded = nhits.unsqueeze(1)
            x = torch.cat([x, nhits_expanded], dim=1)

        # Already max pooled over points so mask is now irrelevant
        # x = self.relu(self.bn1(self.fc1(x)))
        # x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.dropout(_layer_bn_relu_mask(x, self.fc1, self.bn1))
        x = self.dropout(_layer_bn_relu_mask(x, self.fc2, self.bn2))
        x = self.fc3(x)

        return x, trans_feat
    
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class PointNetLoss(torch.nn.Module):
    def __init__(self, loss_fn, mat_diff_loss_scale=DEFAULT_MAT_DIFF_SCALE):
        super().__init__()
        self.loss_fn = loss_fn
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, logits, target, trans_feat=None):
        base_loss = self.loss_fn(logits, target)

        if trans_feat is None or self.mat_diff_loss_scale == 0.0:
            return base_loss

        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = base_loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    aaa0 = torch.ones([2, 4, 4])
    aaa2 = torch.ones([2, 4, 2])

    aaa1 = torch.ones_like(aaa0)
    aaa1[1, :, 3:] = 0

    torch.manual_seed(0)

    model = PointNet()
    model.eval()

    model0 = PointNetEncoder(global_feat=True, feature_transform=False, channel=4)
    model0.eval()

    tmp0, tmp1 = model(aaa0)

    mask = torch.ones([2, 4], dtype=torch.int)
    mask[1, 2:] = 0

    tmp2, tmp3 = model(aaa1, mask=mask)
    tmp4, tmp5 = model(aaa2)

    print(tmp0.shape)
    print(tmp1.shape)

    print(tmp0)
    print(tmp2)
    print(tmp4)




import PIL.Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np
import scipy.stats as st
import gc
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from tap import Tap
import glob
from transformers import AutoModel

# Model expects 26 channels - this is fixed by the pretrained TimeSformer architecture
# If your scroll has fewer layers, they will be padded with zeros
IN_CHANS = 26


class InferenceArgumentParser(Tap):
    # Change based on scroll
    segment_path: str = './train_scrolls/scroll5'
    out_path: str = "./outputs/scroll5"
    start_idx: int = 0
    end_idx: int = 20

    segment_id: list[str] = []
    stride: int = 2
    workers: int = 4
    batch_size: int = 64
    size: int = 64
    device: str = 'cuda'


args = InferenceArgumentParser().parse_args()

PIL.Image.MAX_IMAGE_PIXELS = 933120000
print(f"Using device: {args.device}")


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


class CFG:
    size = 64
    tile_size = 64
    stride = tile_size // 3
    valid_batch_size = 256
    num_workers = 4
    seed = 42

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=tuple([0.0] * IN_CHANS),
            std=tuple([1.0] * IN_CHANS)
        ),
        ToTensorV2(transpose_mask=True),
    ]


def set_seed(seed, cudnn_deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def cfg_init(cfg):
    set_seed(cfg.seed)


cfg_init(CFG)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


def read_image_mask(fragment_id, start_idx, end_idx):
    images = []
    idxs = range(start_idx, end_idx)
    actual_layers_read = 0

    for i in idxs:
        fragment_path = f"{args.segment_path}/{fragment_id}/layers/{i:02}"
        if os.path.exists(f"{fragment_path}.tif"):
            image = cv2.imread(f"{fragment_path}.tif", 0)
            actual_layers_read += 1
        elif os.path.exists(f"{fragment_path}.jpg"):
            image = cv2.imread(f"{fragment_path}.jpg", 0)
            actual_layers_read += 1
        else:
            # If layer doesn't exist, create a zero layer
            if len(images) > 0:
                image = np.zeros_like(images[0])
            else:
                # Default size if no previous images
                image = np.zeros((1000, 1000), dtype=np.uint8)

        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image, 0, 200)
        images.append(image)
    images = np.stack(images, axis=2)

    # Always ensure we have exactly IN_CHANS channels
    if images.shape[2] < IN_CHANS:
        pad_channels = IN_CHANS - images.shape[2]
        padding = np.zeros((images.shape[0], images.shape[1], pad_channels))
        images = np.concatenate([images, padding], axis=2)
        print(
            f"Padded {actual_layers_read} actual layers to {IN_CHANS} channels for model compatibility")

    # Handle reversed segments
    if any(id_ in fragment_id for id_ in ['20230701020044', 'verso', '20230901184804', '20230901234823', '20230531193658', '20231007101615', '20231005123333', '20231011144857', '20230522215721', '20230919113918', '20230625171244', '20231022170900', '20231012173610', '20231016151000']):
        print("Reverse Segment")
        images = images[:, :, ::-1]

    # Load fragment mask
    fragment_mask = None
    wildcard_path_mask = f'{args.segment_path}/{fragment_id}/*_mask.png'
    if os.path.exists(f'{args.segment_path}/{fragment_id}/{fragment_id}_mask.png'):
        fragment_mask = cv2.imread(
            f"{args.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
        fragment_mask = np.pad(
            fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    elif len(glob.glob(wildcard_path_mask)) > 0:
        mask_path = glob.glob(wildcard_path_mask)[0]
        fragment_mask = cv2.imread(mask_path, 0)
        fragment_mask = np.pad(
            fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        # White mask
        fragment_mask = np.ones_like(images[:, :, 0]) * 255

    return images, fragment_mask


def get_img_splits(fragment_id, s, e):
    images = []
    xyxys = []
    if not os.path.exists(f"{args.segment_path}/{fragment_id}"):
        fragment_id = fragment_id + "_superseded"
    print('Reading ', fragment_id)

    try:
        image, fragment_mask = read_image_mask(fragment_id, s, e)
    except Exception as e:
        print("Aborted reading fragment", fragment_id, e)
        return None

    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2] == 0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])

    test_dataset = CustomDatasetTest(images, np.stack(xyxys), CFG, transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean=tuple([0.0] * IN_CHANS),
            std=tuple([1.0] * IN_CHANS)
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.valid_batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers,
                             pin_memory=True,
                             drop_last=False)
    return test_loader, np.stack(xyxys), (image.shape[0], image.shape[1]), fragment_mask


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        return image, xy


class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        print("Loading TimeSformer model from Hugging Face...")
        # Note: The pretrained model expects 26 channels, which matches our IN_CHANS constant
        # If your scroll has fewer layers, they will be padded with zeros automatically
        self.backbone = AutoModel.from_pretrained(
            "scrollprize/timesformer_GP_scroll1",
            trust_remote_code=True
        )
        print("Model loaded successfully!")

    def forward(self, x):
        # The Hugging Face model expects input in the format (B, 1, C, 64, 64) where C=IN_CHANS
        x = self.backbone(x)
        return x


def predict_fn(test_loader, model, device, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    mask_count_kernel = np.ones((CFG.size, CFG.size))
    kernel = gkern(CFG.size, 1)
    kernel = kernel / kernel.max()
    model.eval()

    # Move the kernel to the GPU
    kernel_tensor = torch.tensor(kernel, device=device)

    for _, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds)

        # Resize all predictions at once
        y_preds_resized = F.interpolate(
            y_preds.float(), scale_factor=16, mode='bilinear')

        # Multiply by the kernel tensor
        y_preds_multiplied = y_preds_resized * kernel_tensor
        y_preds_multiplied = y_preds_multiplied.squeeze(1)
        # Move results to CPU as NumPy array
        y_preds_multiplied_cpu = y_preds_multiplied.cpu().numpy()

        # Update mask_pred and mask_count
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += y_preds_multiplied_cpu[i]
            mask_count[y1:y2, x1:x2] += mask_count_kernel

    mask_pred /= np.clip(mask_count, a_min=1, a_max=None)
    return mask_pred


if __name__ == "__main__":
    # Initialize model
    print("Initializing model...")
    model = InferenceModel()
    model.to(device)
    model.eval()

    # Set up segments
    if len(args.segment_id) == 0:
        args.segment_id = [os.path.basename(x) for x in glob.glob(
            f"{args.segment_path}/*") if os.path.isdir(x)]
        args.segment_id.sort()
        print(f"Found {len(args.segment_id)} segments: {args.segment_id}")

    try:
        for fragment_id in args.segment_id:
            try:
                img_split = get_img_splits(
                    fragment_id, args.start_idx, args.end_idx
                )
                if img_split is None:
                    continue
                test_loader, test_xyxz, test_shape, fragment_mask = img_split
                mask_pred = predict_fn(test_loader, model, device, test_shape)
                mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
                mask_pred /= mask_pred.max()

                if len(args.out_path) > 0:
                    # Save as image
                    image_cv = (mask_pred * 255).astype(np.uint8)
                    try:
                        os.makedirs(args.out_path, exist_ok=True)
                    except:
                        pass
                    cv2.imwrite(os.path.join(
                        args.out_path, f"{fragment_id}_prediction.png"), image_cv)
                    print(f"Saved prediction for {fragment_id}")

                del mask_pred
                print(f"Successfully processed {fragment_id}")
                gc.collect()
            except Exception as e:
                print(f"Failed to process {fragment_id}: {e}")
    except Exception as e:
        print(f"Final Exception: {e}")
    finally:
        try:
            del test_loader
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        del model
        torch.cuda.empty_cache()

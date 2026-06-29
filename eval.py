import os, sys
import argparse
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import TestSetLoader
from metrics import mIoU, SamplewiseSigmoidMetric, PD_FA
from model.FGENet import MFENet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description="FGENet Evaluation")
    parser.add_argument("--weight_nuaa", type=str, default="",
                        help="Path to NUAA-SIRST checkpoint")
    parser.add_argument("--weight_irstd", type=str, default="",
                        help="Path to IRSTD-1K checkpoint")
    parser.add_argument("--weight_nudt", type=str, default="",
                        help="Path to NUDT-SIRST checkpoint")
    parser.add_argument("--energy_ratios", type=float, nargs=4,
                        default=[0.8, 0.6, 0.4, 0.1],
                        help="Energy ratios for FFT pyramid")
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), 'dataset'),
                        help="Dataset root directory")
    return parser.parse_args()


def test_one(ds_name, weight_path, energy_ratios, dataset_dir):
    test_set = TestSetLoader(dataset_dir, ds_name, ds_name, img_norm_cfg=None)
    test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)

    net = MFENet(Train=False, energy_ratios=energy_ratios).to(device)
    state = torch.load(weight_path, map_location=device)
    new_sd = OrderedDict()
    for k, v in state['state_dict'].items():
        new_sd[k[6:] if k.startswith('model.') else k] = v
    net.load_state_dict(new_sd)
    net.eval()

    iou_metric = mIoU()
    niou_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
    pd_fa = PD_FA()
    with torch.no_grad():
        for img, gt_mask, size, _ in tqdm(test_loader, desc=ds_name, leave=False):
            img, gt_mask = img.to(device), gt_mask.to(device)
            pred = net(img)
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]

            iou_metric.update((pred > 0.5), gt_mask)
            niou_metric.update(pred, gt_mask)
            pd_fa.update((pred[0, 0] > 0.5).cpu(), gt_mask[0, 0], size)

    pixAcc, mIOU = iou_metric.get()
    nIoU = niou_metric.get()
    Pd, Fa = pd_fa.get()

    return {
        'pixAcc': pixAcc * 100,
        'mIoU': mIOU * 100,
        'nIoU': nIoU * 100,
        'Pd': Pd * 100,
        'Fa': Fa * 1e6,
    }


# ── 主循环 ──────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()

    CONFIGS = [
        dict(name='NUAA-SIRST', weight=args.weight_nuaa),
        dict(name='IRSTD-1K', weight=args.weight_irstd),
        dict(name='NUDT-SIRST', weight=args.weight_nudt),
    ]

    # Filter out configs with empty weight
    CONFIGS = [c for c in CONFIGS if c['weight']]

    if not CONFIGS:
        print("Error: No weight paths provided. Please specify at least one of:")
        print("  --weight_nuaa PATH   --weight_irstd PATH   --weight_nudt PATH")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"{'Dataset':<16} {'mIoU':>8} {'nIoU':>8} {'Pd':>8} {'Fa(×10⁶)':>10} {'pixAcc':>8}")
    print(f"{'-'*70}")
    all_results = {}
    for cfg in CONFIGS:
        results = test_one(cfg['name'], cfg['weight'], args.energy_ratios, args.dataset_dir)
        all_results[cfg['name']] = results
        print(f"{cfg['name']:<16} {results['mIoU']:>7.2f}% {results['nIoU']:>7.2f}% "
              f"{results['Pd']:>7.2f}% {results['Fa']:>10.4f} {results['pixAcc']:>7.2f}%")
    print(f"{'='*70}")
    print("Done.")

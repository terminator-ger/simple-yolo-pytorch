"""
Test script to check if loss function produces NaN values
"""

import torch
import sys
sys.path.insert(0, '/home/michael/data/dev/simple-yolo-pytorch')

from configs.my_config import MyConfig
from yolo.core.loss.yolo_loss import get_loss_fn

def test_loss_nan():
    config = MyConfig()
    config.init_dependent_config()  # Initialize dependent configs
    loss_fn = get_loss_fn(config)
    
    # Create dummy predictions (3 scales: 64x64, 32x32, 16x16) - matches 512x512 image
    batch_size = 2
    num_anchor = 3
    num_class = config.num_class
    
    # Predictions for each scale (img_size=512, downsample=[8,16,32])
    preds = []
    grid_sizes = [(64, 64), (32, 32), (16, 16)]
    
    for h, w in grid_sizes:
        # Shape: (batch, anchor, h, w, 5+num_class)
        pred = torch.randn(batch_size, num_anchor, h, w, 5 + num_class, dtype=torch.float32)
        preds.append(pred)
    
    # Create dummy bboxes: (batch_idx, x, y, w, h)
    # Format: [batch_idx, x, y, w, h] - center coordinates and size, all normalized to [0,1]
    bboxes = torch.tensor([
        [0, 0.3, 0.3, 0.2, 0.2],
        [1, 0.7, 0.7, 0.15, 0.15],
    ], dtype=torch.float32)
    
    # Classes: (batch_idx, class_id)
    classes = torch.tensor([
        [0, 0],
        [1, 1],
    ], dtype=torch.float32)
    
    print("Testing loss function...")
    print(f"Predictions shapes: {[p.shape for p in preds]}")
    print(f"Bboxes shape: {bboxes.shape}")
    print(f"Classes shape: {classes.shape}")
    
    # Test multiple forward passes
    for i in range(20):
        try:
            loss, (conf_loss, iou_loss, class_loss) = loss_fn(preds, bboxes, classes)
            
            print(f"\nIteration {i+1}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Conf Loss: {conf_loss:.6f}")
            print(f"  IoU Loss: {iou_loss:.6f}")
            print(f"  Class Loss: {class_loss:.6f}")
            
            if torch.isnan(loss):
                print("  ❌ NaN detected in loss!")
                return False
            
            print("  ✓ No NaN")
        except Exception as e:
            print(f"\n❌ Error at iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✅ All tests passed! Loss is stable.")
    return True

if __name__ == '__main__':
    success = test_loss_nan()
    sys.exit(0 if success else 1)

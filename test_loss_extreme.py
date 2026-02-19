"""
Comprehensive test script to check if loss function handles extreme cases
"""

import torch
import sys
sys.path.insert(0, '/home/michael/data/dev/simple-yolo-pytorch')

from configs.my_config import MyConfig
from yolo.core.loss import get_loss_fn

def test_loss_extreme_cases():
    config = MyConfig()
    config.init_dependent_config()
    loss_fn = get_loss_fn(config)
    
    batch_size = 2
    num_anchor = 3
    num_class = config.num_class
    
    # Predictions for each scale
    grid_sizes = [(64, 64), (32, 32), (16, 16)]
    
    test_cases = [
        ("Normal values", lambda: torch.randn(batch_size, num_anchor, 64, 64, 7), 
                              lambda: torch.randn(batch_size, num_anchor, 32, 32, 7),
                              lambda: torch.randn(batch_size, num_anchor, 16, 16, 7)),
        
        ("Very large values", lambda: torch.randn(batch_size, num_anchor, 64, 64, 7) * 100, 
                           lambda: torch.randn(batch_size, num_anchor, 32, 32, 7) * 100,
                           lambda: torch.randn(batch_size, num_anchor, 16, 16, 7) * 100),
        
        ("Very small values", lambda: torch.randn(batch_size, num_anchor, 64, 64, 7) * 0.001, 
                           lambda: torch.randn(batch_size, num_anchor, 32, 32, 7) * 0.001,
                           lambda: torch.randn(batch_size, num_anchor, 16, 16, 7) * 0.001),
        
        ("Mixed values", lambda: torch.cat([torch.randn(batch_size, num_anchor, 64, 64, 5), 
                                           torch.ones(batch_size, num_anchor, 64, 64, 2) * -100], dim=-1), 
                     lambda: torch.cat([torch.randn(batch_size, num_anchor, 32, 32, 5), 
                                       torch.ones(batch_size, num_anchor, 32, 32, 2) * 100], dim=-1),
                     lambda: torch.cat([torch.randn(batch_size, num_anchor, 16, 16, 5), 
                                       torch.ones(batch_size, num_anchor, 16, 16, 2) * 1000], dim=-1)),
    ]
    
    # Bboxes 
    bboxes = torch.tensor([
        [0, 0.3, 0.3, 0.2, 0.2],
        [1, 0.7, 0.7, 0.15, 0.15],
    ], dtype=torch.float32)
    
    classes = torch.tensor([
        [0, 0],
        [1, 1],
    ], dtype=torch.float32)
    
    print("Testing loss function with extreme cases...")
    
    all_passed = True
    for test_name, pred1_fn, pred2_fn, pred3_fn in test_cases:
        try:
            preds = [pred1_fn(), pred2_fn(), pred3_fn()]
            loss, (conf_loss, iou_loss, class_loss) = loss_fn(preds, bboxes, classes)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ {test_name}: Loss is NaN/Inf!")
                all_passed = False
            else:
                print(f"✓ {test_name}: Loss = {loss.item():.6f}")
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
            all_passed = False
    
    # Test with empty bboxes
    print("\nTesting with empty bboxes...")
    try:
        preds = [torch.randn(batch_size, num_anchor, 64, 64, 7),
                 torch.randn(batch_size, num_anchor, 32, 32, 7),
                 torch.randn(batch_size, num_anchor, 16, 16, 7)]
        loss, (conf_loss, iou_loss, class_loss) = loss_fn(preds, torch.tensor([], dtype=torch.float32).reshape(0, 5), 
                                                          torch.tensor([], dtype=torch.float32).reshape(0, 2))
        
        if torch.isnan(loss):
            print(f"❌ Empty bboxes: Loss is NaN!")
            all_passed = False
        else:
            print(f"✓ Empty bboxes: Loss = {loss.item():.6f}")
    except Exception as e:
        print(f"❌ Empty bboxes: Error - {e}")
        all_passed = False
    
    if all_passed:
        print("\n✅ All extreme case tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return all_passed

if __name__ == '__main__':
    success = test_loss_extreme_cases()
    sys.exit(0 if success else 1)

import torch
from pathlib import Path

INPUT_DIM = 768
path = Path("app/models/artifacts/classifier_head.pt")
path.parent.mkdir(parents=True, exist_ok=True)

model = torch.nn.Sequential(
    torch.nn.LayerNorm(INPUT_DIM),
    torch.nn.Linear(INPUT_DIM, 256),
    torch.nn.GELU(),
    torch.nn.Dropout(p=0.15),
    torch.nn.Linear(256, 2),
)

with torch.no_grad():
    torch.manual_seed(42)
    for _, param in model.named_parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

torch.save(model.state_dict(), path)
print(f"Saved {path.resolve()} with shape {[tuple(p.shape) for p in model.state_dict().values()]}")

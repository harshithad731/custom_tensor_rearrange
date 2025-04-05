import numpy as np
import einops
from sarvam import rearrange

x = np.random.rand(3, 4)
print(np.allclose(rearrange(x, 'h w -> w h'), einops.rearrange(x, 'h w -> w h')))

x = np.random.rand(12, 10)
print(np.allclose(rearrange(x, '(h w) c -> h w c', h=3), einops.rearrange(x, '(h w) c -> h w c', h=3)))

x = np.random.rand(3, 4, 5)
print(np.allclose(rearrange(x, 'a b c -> (a b) c'), einops.rearrange(x, 'a b c -> (a b) c')))

x = np.random.rand(3, 1, 5)
try:
  print(np.allclose(rearrange(x, 'a 1 c -> a b c', b=4), einops.rearrange(x, 'a 1 c -> a b c', b=4)))
except Exception as e:
  print(e)

x = np.random.rand(2, 3, 4, 5)
print(np.allclose(rearrange(x, '... h w -> ... (h w)'), einops.rearrange(x, '... h w -> ... (h w)')))

x = np.random.rand(2, 3, 4)
print(np.allclose(rearrange(x, 'b c d -> b (c d)'), einops.rearrange(x, 'b c d -> b (c d)')))

x = np.random.rand(2, 12)
print(np.allclose(rearrange(x, 'b (c d) -> b c d', c=3, d=4), einops.rearrange(x, 'b (c d) -> b c d', c=3, d=4)))

x = np.random.rand(2, 3, 4)
print(np.allclose(rearrange(x, 'b c d -> d b c'), einops.rearrange(x, 'b c d -> d b c')))

x = np.random.rand(10, 20)
print(np.allclose(rearrange(x, 'h w -> h w 1'), einops.rearrange(x, 'h w -> h w 1')))

x = np.random.rand(2, 3, 4, 5)
print(np.allclose(rearrange(x, 'b c d e -> b (c d e)'), einops.rearrange(x, 'b c d e -> b (c d e)')))

x = np.random.rand(2, 3, 4)
print(np.allclose(rearrange(x, '... d -> d ...'), einops.rearrange(x, '... d -> d ...')))

x = np.random.rand(12, 10)
print(np.allclose(rearrange(x, '(h w) c -> h w c', h=3), einops.rearrange(x, '(h w) c -> h w c', h=3)))


import pyproj

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
x, y = transformer.transform(116.41493967778204, 39.8988717849168)
print(f"X: {x:.6f}, Y: {y:.6f}")
# 输出: X: 444927.423, Y: 4417296.668
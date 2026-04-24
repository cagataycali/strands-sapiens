# Example: end-to-end segmentation on a real human image.
# Matches the verified run on /tmp/sapiens2_test/input/human.jpg.

from strands_sapiens import sapiens_info, sapiens_seg

print(sapiens_info())

result = sapiens_seg(
    input_path="/tmp/sapiens2_test/input",
    output_dir="/tmp/sapiens2_test/output",
    model_size="0.4b",
    save_pred=True,
)
print(result)

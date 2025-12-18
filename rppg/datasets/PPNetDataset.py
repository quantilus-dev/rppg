class PPNetDataset(Dataset):
    def __init__(self, video_data, label_data):
        # video_data: List of tensors or arrays (N, T, H, W, C)
        # label_data: List of dicts or arrays with {'sbp': ..., 'dbp': ...} or similar
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.video_data = video_data
        self.label_data = label_data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Load Video: (T, H, W, C) -> (C, T, H, W) or matching Model input
        # EfficientPhys uses (C, H, W) per frame, let's stick to (C, T, H, W) for 3D CNN
        video = torch.tensor(self.video_data[index], dtype=torch.float32)
        # Transpose: (T, H, W, C) -> (C, T, H, W)
        video = video.permute(3, 0, 1, 2)

        # Load Label: SBP/DBP
        # Assuming label_data[index] contains [sbp, dbp] or just scalar signal
        # For now, let's assume it's a tensor of shape (T, 2) or (2,)
        label = torch.tensor(self.label_data[index], dtype=torch.float32)

        if torch.cuda.is_available():
            video = video.to('cuda')
            label = label.to('cuda')

        return video, label

    def __len__(self):
        return len(self.video_data)

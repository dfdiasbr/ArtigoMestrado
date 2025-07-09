import timm

def unlock_layers(model_name):
    # Load the model using timm
    model = timm.create_model(model_name, pretrained=True)

    # Layer unlocking counter
    num_layers_unlocked = 0

    if model_name == 'vit_base_patch16_384':
        print(f"Unlocking layers for model {model_name}...")
        # Unlock the last block (block 11) and subsequent layers
        for name, param in model.named_parameters():
            if 'blocks.11' in name or 'norm' in name or 'head' in name:
                param.requires_grad = True
                num_layers_unlocked += 1
            else:
                param.requires_grad = False
        print(f"Layers unlocked: {num_layers_unlocked}")

    elif model_name in ['resnet50', 'resnet101']:
        print(f"Unlocking layers for model {model_name}...")
        # Unlock Layer4 and Fully Connected (fc)
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
                num_layers_unlocked += 1
            else:
                param.requires_grad = False
        print(f"Layers unlocked: {num_layers_unlocked}")

    elif model_name == 'convnext_base':
        print(f"Unlocking layers for model {model_name}...")
        # Unlock blocks in the fourth stage (stages.3.blocks) and subsequent layers
        for name, param in model.named_parameters():
            if 'stages.3.blocks' in name or 'norm' in name or 'head' in name:
                param.requires_grad = True
                num_layers_unlocked += 1
            else:
                param.requires_grad = False
        print(f"Layers unlocked: {num_layers_unlocked}")

    elif model_name == 'coatnet_2_rw_224.sw_in12k':
        print(f"Unlocking layers for model {model_name}...")
        # Unlock blocks from the fourth stage (stages.3.blocks), fifth stage (stages.4.blocks), and subsequent layers
        for name, param in model.named_parameters():
            if 'stages.3.blocks' in name or 'stages.4.blocks' in name or 'norm' in name or 'head' in name:
                param.requires_grad = True
                num_layers_unlocked += 1
            else:
                param.requires_grad = False
        print(f"Layers unlocked: {num_layers_unlocked}")

    else:
        # Unlock the last 10 layers for any other model
        print(f"Unlocking the last 10 layers for model {model_name}...")
        total_layers = len([p for p in model.parameters()])
        unlocked_layers_count = 0
        
        for idx, (name, param) in enumerate(reversed(list(model.named_parameters()))):
            if unlocked_layers_count < 10:
                param.requires_grad = True
                num_layers_unlocked += 1
                unlocked_layers_count += 1
            else:
                param.requires_grad = False

        print(f"Layers unlocked: {num_layers_unlocked}")

    return model

# Example usage
model_name = 'xception'
model = unlock_layers(model_name)

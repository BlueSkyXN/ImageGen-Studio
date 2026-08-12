import asyncio
import execution
import server
from nodes import (
    init_extra_nodes, CheckpointLoaderSimple, EmptyLatentImage, KSampler,
    VAEDecode, SaveImage, NODE_CLASS_MAPPINGS, LoadImage, VAEEncode,
    VAEEncodeForInpaint, ImagePadForOutpaint, LatentUpscaleBy, RepeatLatentBatch
)


def import_custom_nodes() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    loop.run_until_complete(init_extra_nodes())

import_custom_nodes()

CLIPTextEncode = NODE_CLASS_MAPPINGS['CLIPTextEncode']
CLIPTextEncodeSDXL = NODE_CLASS_MAPPINGS['CLIPTextEncodeSDXL']
LoraLoader = NODE_CLASS_MAPPINGS['LoraLoader']
CLIPSetLastLayer = NODE_CLASS_MAPPINGS['CLIPSetLastLayer']

if 'EmptyHunyuanImageLatent' in NODE_CLASS_MAPPINGS:
    EmptyHunyuanImageLatent = NODE_CLASS_MAPPINGS['EmptyHunyuanImageLatent']
else:
    print("⚠️ Warning: 'EmptyHunyuanImageLatent' not found in NODE_CLASS_MAPPINGS. HunyuanImage txt2img may fail if this node is required.")

try:
    KSamplerNode = NODE_CLASS_MAPPINGS['KSampler']
    SAMPLER_CHOICES = KSamplerNode.INPUT_TYPES()["required"]["sampler_name"][0]
    SCHEDULER_CHOICES = KSamplerNode.INPUT_TYPES()["required"]["scheduler"][0]
except Exception:
    print("⚠️ Could not dynamically get sampler/scheduler choices, using fallback list.")
    SAMPLER_CHOICES = ['euler', 'dpmpp_2m_sde_gpu']
    SCHEDULER_CHOICES = ['normal', 'karras']

checkpointloadersimple = CheckpointLoaderSimple()
loraloader = LoraLoader()


print("✅ ComfyUI custom nodes and class mappings are ready.")

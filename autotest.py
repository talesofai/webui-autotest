import csv
from io import StringIO
from PIL import Image
import numpy as np
import os
import math
import random
from copy import copy

import modules.scripts as scripts
import gradio as gr

from modules import images, sd_samplers, processing, sd_models, sd_vae, sd_samplers_kdiffusion
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
import modules.shared as shared
import modules.sd_samplers
import modules.sd_models
import modules.sd_vae
import re

from modules.ui_components import ToolButton

class SharedSettingsStackHelper(object):
    def __enter__(self):
        self.CLIP_stop_at_last_layers = opts.CLIP_stop_at_last_layers
        self.vae = opts.sd_vae
        self.uni_pc_order = opts.uni_pc_order

    def __exit__(self, exc_type, exc_value, tb):
        opts.data["sd_vae"] = self.vae
        opts.data["uni_pc_order"] = self.uni_pc_order
        modules.sd_models.reload_model_weights()
        modules.sd_vae.reload_vae_weights()

        opts.data["CLIP_stop_at_last_layers"] = self.CLIP_stop_at_last_layers



class Script(scripts.Script):
    def title(self):
        return "autotest"

    def ui(self, is_img2img):
        gr.HTML('<br />')


        with gr.Row():
            with gr.Column():

                with gr.Row():

                    character_test_activate = gr.Checkbox(label='character test case', value=True,
                                           elem_id=self.elem_id("character_test_activate"))

                    style_test_activate = gr.Checkbox(label='style test case', value=False,
                                                      elem_id=self.elem_id("style_test_activate"))

                with gr.Row():
                    complex_checkpoint_test = gr.Checkbox(label='complex checkpoint test case', value=False,
                                           elem_id=self.elem_id("complex_checkpoint_test"))
                    complex_checkpoint_values = gr.Dropdown(label="checkpoint names",
                                                    choices=sorted(sd_models.checkpoints_list, key=str.casefold),
                                                    multiselect=True,
                                                    interactive=True, elem_id=self.elem_id("complex_checkpoint_values"))

                with gr.Row():
                    simple_test_activate = gr.Checkbox(label='some simple test case', value=True,
                                           elem_id=self.elem_id("simple_test_activate"))
                    select_test_activate = gr.Checkbox(label='some selected test case', value=True,
                                                       elem_id=self.elem_id("select_test_activate"))

                    # enable_input_values = gr.Checkbox(label='activate input test cases', value=True,
                    #                        elem_id=self.elem_id("enable_input_values"))
                # with gr.Row():
                #     a_check = gr.Checkbox(label="a activated", value=True, elem_id=self.elem_id("a_check"))
                #     a_values = gr.Textbox(label="a values", lines=1, elem_id=self.elem_id("a_values"))
                #
                # with gr.Row():
                #     b_check = gr.Checkbox(label="b activated", value=True, elem_id=self.elem_id("b_check"))
                #     b_values = gr.Textbox(label="b values", lines=1, elem_id=self.elem_id("b_values"))
                #
                # with gr.Row():
                #     c_check = gr.Checkbox(label="c activated", value=True, elem_id=self.elem_id("c_check"))
                #     c_values = gr.Textbox(label="c values", lines=1, elem_id=self.elem_id("c_values"))

                #with gr.Row():
                #    d_check = gr.Checkbox(label="d activated", value=True, elem_id=self.elem_id("d_check"))
                #    d_values = gr.Textbox(label="d values", lines=1, elem_id=self.elem_id("d_values"))

                #complex_test_activate = gr.Checkbox(label='some complex test case', value=False,
                #                          elem_id=self.elem_id("complex_test_activate"))

                with gr.Row():
                    checkpoint_test_activate = gr.Checkbox(label='some checkpoint test case', value=False,
                                                  elem_id=self.elem_id("checkpoint_test_activate"))
                    checkpoint_values = gr.Dropdown(label="checkpoint names",
                                                    choices=sorted(sd_models.checkpoints_list, key=str.casefold), multiselect=True,
                                                    interactive=True, elem_id=self.elem_id("checkpoint_values"))

                with gr.Row():
                    lora_test_activate = gr.Checkbox(label='some lora test case', value=False,
                                            elem_id=self.elem_id("lora_test_activate"))
                    lora_values = gr.Textbox(label="lora names", lines=1, elem_id=self.elem_id("lora_values"))

                margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2,
                                        elem_id=self.elem_id("margin_size"))

                #put_at_start = gr.Checkbox(label='Put variable parts at start of prompt', value=False, elem_id=self.elem_id("put_at_start"))
                #different_seeds = gr.Checkbox(label='Use different seed for each picture', value=False, elem_id=self.elem_id("different_seeds"))
            '''
            with gr.Column():
                prompt_type = gr.Radio(["positive", "negative"], label="Select prompt", elem_id=self.elem_id("prompt_type"), value="positive")
                variations_delimiter = gr.Radio(["comma", "space"], label="Select joining char", elem_id=self.elem_id("variations_delimiter"), value="comma")
            
            enable_input_values, a_check, a_values, b_check, b_values, c_check, c_values,
            enable_input_values, a_check, a_values, b_check, b_values, c_check, c_values,
            '''
        return [character_test_activate, style_test_activate, complex_checkpoint_test, complex_checkpoint_values, simple_test_activate, select_test_activate, checkpoint_test_activate, checkpoint_values, lora_test_activate, lora_values, margin_size]
        #return [put_at_start, different_seeds, prompt_type, variations_delimiter, margin_size]

    def run(self, p, character_test_activate, style_test_activate, complex_checkpoint_test, complex_checkpoint_values, simple_test_activate, select_test_activate, checkpoint_test_activate, checkpoint_values, lora_test_activate, lora_values, margin_size):#put_at_start, different_seeds, prompt_type, variations_delimiter, margin_size):
        # Raise error if promp type is not positive or negative
        #if prompt_type not in ["positive", "negative"]:
        #    raise ValueError(f"Unknown prompt type {prompt_type}")
        # Raise error if variations delimiter is not comma or space
        #if variations_delimiter not in ["comma", "space"]:
        #    raise ValueError(f"Unknown variations delimiter {variations_delimiter}")




        def simple_test(pc ,default_y_labels, simple_test_activate, margin_size, lora_name):
            p = copy(pc)
            modules.processing.fix_seed(p)

            if not simple_test_activate:
                return Processed(p, [], p.seed, "")

            size = ['full body', 'half body']
            camera = ['from above', 'from below']
            #camera = ['profile', 'from above', 'from below', 'close up']
            activity = ['stand' ,'sit']
            #activity = ['stand', 'sit', 'run', 'clenched hands', 'pray', 'dynamic pose']
            emoji = ['666']
            #emoji = ['😃','😭','😮']



            #lora_weight = ['0.2','0.4','0.6','0.8','1']
            lora_weight = ['0.8']

            delimiter = ", "

            x_labels = lora_weight
            y_labels = []
            #default_y_labels = ['full body, stand', 'half body, sit', 'profile, run', 'clenched hands, from above', 'Kneel, from below', 'full body, dynamic pose', 'Close-up, pray']
            #default_y_labels = ['half body']
            #default_y_labels = ['walking','arms crossed', 'own hands together', 'arms up', 'knees to chest', 'pointing at self', 'grin', 'crying, tears', '>o<']



            prompt = p.prompt
            # prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
            original_prompt = prompt[0] if type(prompt) == list else prompt
            positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

            #prompt_matrix_parts = original_prompt.split("|")



            #if enable_input_values:
            '''
                            for s in size:
                                for c in camera:
                                    for a in activity:
                                        for e in emoji:
                                            y_labels.append(delimiter.join([s, c, a, e]))
            '''
            #else:
            y_labels = default_y_labels

            combination_count = len(y_labels)

            all_prompts = []
            all_seed = []

            random_num = 4
            random_seed = [p.seed]

            #combination_count = combination_count * (len(random_seed))

            for i in range(random_num):
                random_seed.append(p.seed + random.randint(-524288, 524287))

            for i in y_labels:
                for seed in random_seed:
                    for weight in lora_weight:
                        prompt = original_prompt

                        #pattern = r'0\.2'
                        #replacement = weight
                        #prompt = re.sub(pattern, replacement, prompt, count=1)

                        selected_prompts = [prompt, i]
                        all_prompts.append(delimiter.join(selected_prompts))
                        all_seed.append(seed)



            '''
            combination_count = 2 ** (len(prompt_matrix_parts) - 1)
            for combination_num in range(combination_count):
                selected_prompts = [text.strip().strip(',') for n, text in enumerate(prompt_matrix_parts[1:]) if
                                    combination_num & (1 << n)]
                selected_prompts = [original_prompt] + selected_prompts

                all_prompts.append(delimiter.join(selected_prompts))
            '''

            p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
            p.do_not_save_grid = True

            print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

            p.prompt = all_prompts

            p.seed = all_seed

            #p.seed = [p.seed + (0) for i in range(len(all_prompts))]

            p.prompt_for_display = positive_prompt

            processed = process_images(p)

            for i in range(combination_count):

                hor_texts = [[images.GridAnnotation(str(y))] for y in random_seed]
                ver_texts = [[images.GridAnnotation(x)] for x in x_labels]

                start_index = (i * len(lora_weight) * len(random_seed)) + i
                end_index = start_index + len(lora_weight) * len(random_seed)

                grid = images.image_grid(processed.images[start_index:end_index], rows=1)

                grid = images.draw_grid_annotations(grid, processed.images[start_index].size[0],
                                                        processed.images[start_index].size[1], hor_texts,
                                                        ver_texts, margin_size)

                processed.images.insert(i, grid)
                processed.all_prompts.insert(i, processed.all_prompts[start_index])
                processed.all_seeds.insert(i, processed.all_seeds[start_index])
                processed.infotexts.insert(i, processed.infotexts[start_index])

                images.save_image(processed.images[i], p.outpath_grids + '/autotest/' + lora_name + '/simple_test/', ver_texts[0],
                                  extension=opts.grid_format,
                                  prompt=original_prompt, seed=processed.seed, grid=True, p=p,forced_filename=lora_name + '_' + y_labels[i] + '_' + str(processed.all_seeds[i]))



            #grid = images.image_grid(processed.images, rows=combination_count)
            #grid = images.draw_grid_annotations(grid, processed.images[0].size[0],
            #                                    processed.images[0].size[1], hor_texts, ver_texts,
            #                                    margin_size)
            #processed.images.insert(0, grid)
            #processed.index_of_first_image = 1
            #processed.infotexts.insert(0, processed.infotexts[0])



            #if opts.grid_save:
            #    images.save_image(processed.images[0], p.outpath_grids+'autotest/'+'simple_test/', "prompt_matrix", extension=opts.grid_format,
            #                      prompt=original_prompt, seed=processed.seed, grid=True, p=p)

            return processed

        def complex_test(pc, simple_test_activate, margin_size):

            p = copy(pc)
            modules.processing.fix_seed(p)

            if not p.override_settings['sd_model_checkpoint']:
                model_name = shared.sd_model.sd_checkpoint_info.name
                lora_name = str(model_name)
            else:
                model_name = p.override_settings['sd_model_checkpoint']
                lora_name = str(model_name)

            if not simple_test_activate:
                return Processed(p, [], p.seed, "")

            default_x_labels = ['1girl','1man','5-year-old girl','5-year-old boy']

            delimiter = ", "

            x_labels = default_x_labels
            default_y_labels = ['walking','arms crossed', 'own hands together', 'arms up', 'knees to chest', 'pointing at self', 'grin', 'crying, tears', '>o<']

            pose = ['hand on own face', 'arms behind back, looking at viewer', 'head back, full body',
                                'armpit peek', 'hand on hip, standing, full body ', 'own hands together,  pray',
                                'outstretched arm', 'sit, crossed legs', 'hand to own mouth, close-up,',
                                'thumbs up', 'leaning forward', 'crossed arms', 'lying, from above', 'looking back',
                                'jumping, full body', 'head rest', 'reading books', 'holding umbrella', 'squat',
                                'kneeling']

            emotion = ['crying, tears', 'grin', '>o<']

            color = ['Black', 'White', 'Green', 'Blue', 'Purple', 'Pink', 'Red', 'Yellow']

            head = ['Hair','Eye']

            upper_body = ['Suit', 'Jacket', 'Shirt', 'vest', 'T-shirt']

            lower_body = ['Pants', 'Skirt', 'jeans']

            small_feature = ['cat hair ornament','mask on head', 'dragon horns', 'over-rim eyewear', 'sunglasses', 'black wings', 'off shoulder', 'tail', 'muscle']

            small_animals = ['Cat', 'Dog', 'Cow', 'Bird', 'Butterfly', 'Horse', 'Panda']

            indoor = ['cafeteria', 'living room', 'kitchen', 'bedroom', 'bathroom', 'dining room', 'gym']

            outdoor = ['garden', 'street', 'village', 'zoo', 'market', 'bridge', 'city']

            nature = ['mountain', 'beach', 'river', 'lake', 'ocean bottom', 'forest', 'bamboo forest', 'sun', 'starry sky', 'full moon', 'in the rain']

            objects = ['vehicle', 'rubbish bin', 'fountain', 'bicycle', 'street lamp']

            buildings = ['castle', 'factory', 'theater', 'stadium', 'train station', 'mall', 'airport', 'hospital', 'skyscraper', 'church']

            clothes = []

            for i in range(10):
                temp = [random.choice(color) + ' ' + head[0], random.choice(color) + ' ' + head[1], random.choice(color) + ' ' + random.choice(upper_body),random.choice(color) + ' ' + random.choice(lower_body)]
                clothes.append(delimiter.join(temp))

            objects = ['street, ' + i for i in objects]

            y_labels = [pose[13:], emotion]#, small_feature, small_animals, indoor, outdoor, nature, objects, buildings, clothes]
            y_names = ['pose', 'emotion', 'small_feature', 'small_animals', 'indoor', 'outdoor', 'nature', 'objects', 'buildings', 'clothes']

            prompt = p.prompt
            # prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
            original_prompt = prompt[0] if type(prompt) == list else prompt
            positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

            all_prompts = []

            for group in y_labels:
                for i in group:
                    for j in x_labels:
                        prompt = original_prompt
                        selected_prompts = [j, i, prompt]
                        all_prompts.append(delimiter.join(selected_prompts))

            combination_count = len(all_prompts)
            p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
            p.do_not_save_grid = True

            print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

            p.prompt = all_prompts

            p.prompt_for_display = positive_prompt

            processed = process_images(p)

            start_index = 0

            for i, group in enumerate(y_labels):
                hor_texts = [[images.GridAnnotation(y)] for y in group]
                ver_texts = [[images.GridAnnotation(x)] for x in x_labels]

                end_index = start_index + len(group) * len(x_labels)

                grid = images.image_grid(processed.images[start_index:end_index], rows=len(group))

                grid = images.draw_grid_annotations(grid, processed.images[start_index].size[0],
                                                    processed.images[start_index].size[1], ver_texts,hor_texts,
                                                    margin_size)

                processed.images.insert(i, grid)
                processed.all_prompts.insert(i, processed.all_prompts[start_index])
                processed.all_seeds.insert(i, processed.all_seeds[start_index])
                processed.infotexts.insert(i, processed.infotexts[start_index])

                images.save_image(processed.images[i], p.outpath_grids + '/autotest/' + lora_name + '/comlex_test/',
                                  ver_texts[0],
                                  extension=opts.grid_format,
                                  prompt=original_prompt, seed=processed.seed, grid=True, p=p,
                                  forced_filename=lora_name + '_' + y_names[i])

                start_index = end_index + 1
            return processed

        def complex_lora_test(pc, simple_test_activate, margin_size):

            p = copy(pc)

            p.seed=int(p.seed)

            modules.processing.fix_seed(p)

            if not p.override_settings['sd_model_checkpoint']:
                model_name = shared.sd_model.sd_checkpoint_info.name
                lora_name = str(model_name)
            else:
                model_name = p.override_settings['sd_model_checkpoint']
                lora_name = str(model_name)

            prompt = p.prompt
            # prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
            original_prompt = prompt[0] if type(prompt) == list else prompt
            positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

            if not simple_test_activate:
                return Processed(p, [], p.seed, "")

            delimiter = ", "

            default_x_labels = ['1girl','1man','5-year-old girl','5-year-old boy']



            x_labels = default_x_labels

            lora_prompts = [
                'masterpiece, best quality, 1girl, solo, long_hair, looking_at_viewer, smile, bangs, skirt, shirt, long_sleeves, hat, dress, bow, holding, closed_mouth, flower, frills, hair_flower, petals, bouquet, holding_flower, center_frills, bonnet, holding_bouquet, flower field, flower field, lineart, monochrome, <lora:animeoutlineV4_16:1>',
                '(masterpiece),(best quality),(ultra-detailed), (full body:1.2), 1girl,chibi,cute, smile, open mouth, flower, outdoors, playing guitar, music, beret, holding guitar, jacket, blush, tree, :3, shirt, short hair, cherry blossoms, green headwear, blurry, brown hair, blush stickers, long sleeves, bangs, headphones, black hair, pink flower, (beautiful detailed face), (beautiful detailed eyes), <lora:blindbox_v1_mix:1>,',
                'shukezouma, negative space, , shuimobysim , <lora:shuV2:0.8>, portrait of a woman standing , willow branches, (masterpiece, best quality:1.2), traditional chinese ink painting, <lora:shuimobysimV3:0.7>, modelshoot style, peaceful, (smile), looking at viewer, wearing long hanfu, hanfu, song, willow tree in background, wuchangshuo,',
                '<lora:genshin_impact_kamisato_ayaka:0.8> (kamisato_ayaka \(genshin impact\):0.8), 1girl, solo',
                '<lora:genshin_impact_raiden_shogun:0.8> raiden shogun \(genshin_impact\), 1girl, solo',
                'masterpiece, highres,  <lora:holding_sign_v1:0.85> holding  sign, blank sign'
            ]

            lora_names = ['Anime Lineart', 'blindbox', 'MoXin', 'kamisato_ayaka', 'raiden_shogun', 'holding sign']

            all_prompts = []
            all_seed = []


            random_num = 4
            random_seed = [p.seed]

            for i in range(random_num):
                random_seed.append(p.seed + random.randint(-524288, 524287))

            for i in lora_prompts:
                for seed in random_seed:
                    all_prompts.append(i)
                    all_seed.append(seed)


            p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
            p.do_not_save_grid = True

            print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

            p.prompt = all_prompts

            p.seed = all_seed

            p.prompt_for_display = positive_prompt

            processed = process_images(p)

            hor_texts = [[images.GridAnnotation(y)] for y in lora_names]
            ver_texts = [[images.GridAnnotation(str(x))] for x in random_seed]

            grid = images.image_grid(processed.images, rows=len(lora_names))

            grid = images.draw_grid_annotations(grid, processed.images[0].size[0],
                                                processed.images[0].size[1], ver_texts, hor_texts,
                                                margin_size)

            images.save_image(grid, p.outpath_grids + '/autotest/' + lora_name + '/comlex_lora_test/',
                              ver_texts[0],
                              extension=opts.grid_format,
                              prompt=original_prompt, seed=processed.seed, grid=True, p=p,
                              forced_filename=lora_name + '_lora')

            processed.images.insert(0, grid)
            processed.all_prompts.insert(0, processed.all_prompts[0])
            processed.all_seeds.insert(0, processed.all_seeds[0])
            processed.infotexts.insert(0, processed.infotexts[0])

            return processed

        def select_ckpt_test(pc, simple_test_activate, margin_size):

            p = copy(pc)

            p.seed=int(p.seed)

            modules.processing.fix_seed(p)

            if not p.override_settings['sd_model_checkpoint']:
                model_name = shared.sd_model.sd_checkpoint_info.name
                lora_name = str(model_name)
            else:
                model_name = p.override_settings['sd_model_checkpoint']
                lora_name = str(model_name)

            prompt = p.prompt
            # prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
            original_prompt = prompt[0] if type(prompt) == list else prompt
            positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

            if not simple_test_activate:
                return Processed(p, [], p.seed, "")

            delimiter = ", "

            selcted_prompts = [
                'octans, sky, star (sky), scenery, starry sky, night, 1girl, night sky, solo, outdoors, signature, building, cloud, milky way, sitting, tree, long hair, city, silhouette, cityscapemasterpiece, best quality, ,halftone,cloud, light_particles, space, sky,water,girl,night',
                'masterpiece, intricate detail,highres,best quality, cowboy shot,extremely intricate,black hair,hat ribbon, portrait,backless dress, (long dress:1.2), very long hair, breasts, bare shoulders, sky, ocean, water, looking at viewer, italian, head tilt, wind, cloud, sunlight, outdoor, 1girl, solo, BREAK, cinematic light,',
                '1man, ((pure white background)), baseball cap, hat, solo, smile, hood down, jacket, hood, headphones, closed mouth, blue hair, long sleeves, fingernails, sleeves past wrists, looking at viewer, multicolored eyes, short hair, hooded jacket, red jacket, hair ornament, object hug, blush,white headwear, upper body, blue eyes, claw pose, heart, black headwear, hand up, bow, hoodie, hairclip,button badge, puffy long sleeves, puffy sleeves, black bow, badge, holding, drawstring, red eyes, star (symbol), multicolored hair, lightning bolt symbol, colorful, looking to the side',
                'masterpiece, best quality, ([balloons:Small planets:0.5]:1.4), (Small_planets inside of balloons:1.4), (lots of colorful Small_planets:1.35) (colorful planets, earth, floating petals, big balloons:1.22), 1 girl, cute face, Full body, sitting, detailed beautiful eyes, bare legs, costume combination, Goddess, perfect body, [nsfw:0.88] (sitting on ice_planet:1.22) (lots of [floting blue Butterflies:floting ice:0.4]:1.22) (detailed light), (an extremely delicate and beautiful), volume light, best shadow,cinematic lighting, Depth of field, dynamic angle, Oily skin,',
                '(Extremely detailed floating city:1.3), ((1 detailed mirage-castle floats above the clouds)and(Detailed floats minarets illusion, detailed floats belfry illusion)and(Highly transparent mirage)and(Detailed view of the base of the Mirage castle)),ultra-detailed,extremely detailed and beautiful, sky and white cloud and sunset glow background,ultra-detailed birds, (floating city in bloom, Floating petals, blooming gardens in the sky,  many vines on the wall:1.2)Imagination, dream, 1 girl, (1 girl at floating city:1.3), (solo), (Super wide Angle lens), (back), behind head, white dress, (full body), (bare feet), (Backless dress), the girl away from camera, Holy Light,',
                '1girl，stand',
                '1girl，read books',
                '1girl，headrest',
                '1boy，walking，street',
                '1boy，crying，tears',
                '<lora:genshin_impact_raiden_shogun:0.8> raiden shogun \(genshin_impact\), 1girl, solo,  looking at viewers, medium shot, arm crossed',
                '<lora:genshinImpact_tartaglia_v10:0.7>, (tartaglia\(genshin impact\):0.6), 1boy, (light brown hair, short hair, blue eyes, mask on head),male focus, solo, face focus, looking at viewers, side view, fighting pose, Small river beneath the snow-capped mountains',
                '<lora:spy_family_anya_v3:0.8> anya \(spy x family\), 1girl, solo, running in campus',
                '<lora:Hoshino Ai\'s Pose 2:0.9> 1girl, ai\'s pose, solo, smile, open mouth, one eye closed, star (symbol), pointing, microphone,',
                '1girl, <lora:v_eyes:1> >_<, closed eyes',
                'best quality,<lora:animetarotV51:1>, magic, symmetrical, highres, tarot card, <lora:Genshin_Card:0.3>,1girl',
                '1girl,masterpiece, detailed face and eyes:1.3,perfect eyes:1.1,best quality, {subject}, smile, watery eyes, ((( hanfu,chinese style))), detailed face, long flowing hair, luxury jewelry, luxury hair accessories, delicate features, street background, <lora:moxin_10:0.3>, <lora:moxinloramoxinassist_20230326:0.2>',
                '1girl, masterpiece, best quality, <lora:teradatera:0.6>, chibi, ',
                '1girl, masterpiece, best quality, depth of field, <lora:LORA13:0.8>,1boy',
                'masterpiece, best quality, depth of field, <lora:LORA13:0.8>,1boy, <lora:genshinImpact_tartaglia_v10:0.7>, (tartaglia\(genshin impact\):0.6), 1boy, (light brown hair, short hair, blue eyes, mask on head),male focus, solo, face focus, <lora:Hoshino Ai\'s Pose 2:0.9>, ai\'s pose, solo, smile, open mouth, one eye closed, star (symbol), pointing, microphone,',
                '1woman, Sideways sitting, Holding a coffee cup,',
                '1woman, Kneeling, laughing happily',
                '1woman, stand, arm crossed, wink',
                '1man, yellow jacket, baggy jeans, lying on one\'s side',
                '1man, wedding suit, short blue hair, dog ears, Holding a bunch of flower',
                '1man, magician, V-shaped hand gesture',
                '5-year-old boy, brown hair, sportswear, Lake under the night sky,full body, Side view',
                '5-year-old boy, Small river beneath the snow-capped mountains, Low angle view, 1cat',
                '5-year-old girl, blue ponytail, green dress decorated with white flowers, Archway in a Chinese courtyard, pink flowers, medium shot',
                '5-year-old girl, Hand on cheek, Bakery, full body, long shot',
                '1 old woman, warm sweater, hat, Terrified, close up',
                '1 old woman, Shy, Archway in a Chinese courtyard, pink flowers',
                '1 old man, scientist, Contemptuous, laboratory',
                '1 old man, teacher, Worried, classroom',
                '1girl, grey hair, long curly, hair, choker, black dress, Holding balloons, Bakery',
                '1girl, pink off-shoulder jacket, black vest, sports jeans shorts, sneakers, Pure white background, 1dog',
                '1girl, gold long braid, hanfu, Archway in a Chinese courtyard, pink flowers, medium shot',
                '1girl, school uniform, ear-length short hair, princess, Holding an umbrella, Low angle view',
                '1girl, magician, Hand on forehead, Top-down lighting',
                '1girl, blue and pink twintail, dancer, Backlight',
                '1 robot, Depth of field',
                '1girl, student, Terrified, Lake under the night sky, full body, long shot',
                '1girl, Shy, Small river beneath the snow-capped mountains, closeup',
                '1cat, running, grassland, closeup',
                '1dog, coffee shop, warm blanket, sleeping',
                '1dinosaur, high trees, Lake under the night sky, eating',
                'Lake under the night sky',
                'Small river beneath the snow-capped mountains, Depth of field',
                'Bakery',
                '2girls, dancing, Living room',
            ]

            selcted_prompts = ['octans, sky, star (sky), scenery, starry sky, night, 1girl, night sky, solo, outdoors, signature, building, cloud, milky way, sitting, tree, long hair, city, silhouette, cityscapemasterpiece, best quality, ,halftone,cloud, light_particles, space, sky,water,girl,night',
                'masterpiece, intricate detail,highres,best quality, cowboy shot,extremely intricate,black hair,hat ribbon, portrait,backless dress, (long dress:1.2), very long hair, breasts, bare shoulders, sky, ocean, water, looking at viewer, italian, head tilt, wind, cloud, sunlight, outdoor, 1girl, solo, BREAK, cinematic light,',
                '1man, ((pure white background)), baseball cap, hat, solo, smile, hood down, jacket, hood, headphones, closed mouth, blue hair, long sleeves, fingernails, sleeves past wrists, looking at viewer, multicolored eyes, short hair, hooded jacket, red jacket, hair ornament, object hug, blush,white headwear, upper body, blue eyes, claw pose, heart, black headwear, hand up, bow, hoodie, hairclip,button badge, puffy long sleeves, puffy sleeves, black bow, badge, holding, drawstring, red eyes, star (symbol), multicolored hair, lightning bolt symbol, colorful, looking to the side',
                'masterpiece, best quality, ([balloons:Small planets:0.5]:1.4), (Small_planets inside of balloons:1.4), (lots of colorful Small_planets:1.35) (colorful planets, earth, floating petals, big balloons:1.22), 1 girl, cute face, Full body, sitting, detailed beautiful eyes, bare legs, costume combination, Goddess, perfect body, [nsfw:0.88] (sitting on ice_planet:1.22) (lots of [floting blue Butterflies:floting ice:0.4]:1.22) (detailed light), (an extremely delicate and beautiful), volume light, best shadow,cinematic lighting, Depth of field, dynamic angle, Oily skin,',
                '(Extremely detailed floating city:1.3), ((1 detailed mirage-castle floats above the clouds)and(Detailed floats minarets illusion, detailed floats belfry illusion)and(Highly transparent mirage)and(Detailed view of the base of the Mirage castle)),ultra-detailed,extremely detailed and beautiful, sky and white cloud and sunset glow background,ultra-detailed birds, (floating city in bloom, Floating petals, blooming gardens in the sky,  many vines on the wall:1.2)Imagination, dream, 1 girl, (1 girl at floating city:1.3), (solo), (Super wide Angle lens), (back), behind head, white dress, (full body), (bare feet), (Backless dress), the girl away from camera, Holy Light,'
                               ]
            selcted_prompts = 4 * selcted_prompts

            selected_names = [str(i+1) for i in range(len(selcted_prompts))]

            all_prompts = []
            all_seed = []

            random_seed = [int(p.seed)]

            for i in selcted_prompts:
                all_prompts.append(delimiter.join([i,original_prompt]))
                all_seed.append(int(p.seed)+random.randint(-524288, 524287))

            p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
            p.do_not_save_grid = True

            print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

            p.prompt = all_prompts

            p.seed = all_seed

            p.prompt_for_display = positive_prompt

            processed = process_images(p)

            for i,selected_name in enumerate(selected_names):
                hor_texts = [[images.GridAnnotation(selected_name)]]
                ver_texts = [[images.GridAnnotation(str(all_seed[i]))]]

                grid = images.image_grid([processed.images[i]], rows=1)

                grid = images.draw_grid_annotations(grid, processed.images[i].size[0],
                                                    processed.images[i].size[1], ver_texts, hor_texts,
                                                    margin_size)

                images.save_image(grid, p.outpath_grids + '/autotest/' + lora_name + '/selected_ckpt_test/',
                                  ver_texts[0],
                                  extension=opts.grid_format,
                                  prompt=original_prompt, seed=processed.seed, grid=True, p=p,
                                  forced_filename=selected_name)

            return processed


        def checkpoint_test(pc, checkpoint_test_activate, checkpoint_values, margin_size, lora_name):

            p = copy(pc)
            modules.processing.fix_seed(p)

            if not checkpoint_values:
                checkpoint_values = sorted(sd_models.checkpoints_list, key=str.casefold)
            if not checkpoint_test_activate:
                return Processed(p, [], p.seed, "")

            delimiter = ','

            lora_weight = ['0.2', '0.4', '0.6', '0.8', '1']

            x_labels = lora_weight
            default_y_labels = ['full body, stand', 'half body, sit', 'profile, run', 'clenched hands, from above',
                                'Kneel, from below', 'full body, dynamic pose', 'Close-up, pray']

            prompt = p.prompt
            original_prompt = prompt[0] if type(prompt) == list else prompt
            positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

            y_labels = default_y_labels

            combination_count = len(y_labels)

            list_size = len(y_labels) * len(lora_weight) * len(checkpoint_values)

            processed_result = None

            def apply_checkpoint(p, x):
                info = modules.sd_models.get_closet_checkpoint_match(x)
                if info is None:
                    raise RuntimeError(f"Unknown checkpoint: {x}")
                p.override_settings['sd_model_checkpoint'] = info.name

            p.do_not_save_grid = True

            p.prompt_for_display = positive_prompt
            for ic, checkpoint in enumerate(checkpoint_values):
                p_checkpoint = copy(p)
                apply_checkpoint(p_checkpoint, checkpoint)
                for iy,y in enumerate(y_labels):
                    for iw,weight in enumerate(lora_weight):
                        idx = iy * len(checkpoint_values) * len(lora_weight) + ic * len(lora_weight) + iw
                        prompt = original_prompt
                        pattern = r'0\.2'
                        replacement = weight
                        prompt = re.sub(pattern, replacement, prompt, count=1)

                        selected_prompts = [prompt, y]
                        p_weight = copy(p_checkpoint)
                        p_weight.prompt = delimiter.join(selected_prompts)

                        processed = process_images(p_weight)

                        if processed_result == None:
                            processed_result = copy(processed)
                            processed_result.images = [None] * list_size
                            processed_result.all_prompts = [None] * list_size
                            processed_result.all_seeds = [None] * list_size
                            processed_result.infotexts = [None] * list_size
                            processed_result.index_of_first_image = 1

                        if processed.images:
                            # Non-empty list indicates some degree of success.
                            processed_result.images[idx] = processed.images[0]
                            processed_result.all_prompts[idx] = processed.prompt
                            processed_result.all_seeds[idx] = processed.seed
                            processed_result.infotexts[idx] = processed.infotexts[0]
                        else:
                            cell_mode = "P"
                            cell_size = (processed_result.width, processed_result.height)
                            if processed_result.images[0] is not None:
                                cell_mode = processed_result.images[0].mode
                                # This corrects size in case of batches:
                                cell_size = processed_result.images[0].size
                            processed_result.images[idx] = Image.new(cell_mode, cell_size)

            for i in range(combination_count):
                hor_texts = [[images.GridAnnotation(x)] for x in x_labels]
                ver_texts = [[images.GridAnnotation(y)] for y in checkpoint_values]

                start_index = (i * len(lora_weight) * len(checkpoint_values)) + i
                end_index = start_index + len(lora_weight) * len(checkpoint_values)

                grid = images.image_grid(processed_result.images[start_index:end_index], rows=len(checkpoint_values))

                grid = images.draw_grid_annotations(grid, processed_result.images[start_index].size[0],
                                                    processed_result.images[start_index].size[1], hor_texts,
                                                    ver_texts, margin_size)

                processed_result.images.insert(i, grid)
                processed_result.all_prompts.insert(i, processed_result.all_prompts[start_index])
                processed_result.all_seeds.insert(i, processed_result.all_seeds[start_index])
                processed_result.infotexts.insert(i, processed_result.infotexts[start_index])

                images.save_image(processed_result.images[i], p.outpath_grids + '/autotest/' + lora_name + '/checkpoint_test/', ver_texts[0],
                                  extension=opts.grid_format,
                                  prompt=original_prompt, seed=processed.seed, grid=True, p=p,forced_filename=lora_name + '_' + y_labels[i])

            return processed_result

        def lora_test(pc, lora_test_activate, lora_values, margin_size, lora_name):

            p = copy(pc)
            modules.processing.fix_seed(p)

            if not lora_test_activate:
                return Processed(p, [], p.seed, "")

            lora1_weight = ['0.2', '0.4', '0.6', '0.8', '1']
            lora2_weight = ['0.2', '0.4', '0.6', '0.8', '1']

            delimiter = ", "

            x_labels = lora1_weight
            y_labels = []

            style = ['<lora:blindbox_v1_mix:0.1>, full body, chibi', '<lora:百花酿:0.1>, flower,baihuaniang', '<lora:bichu-v0612:0.1>,bichu,oil painting,Impressionism']
            face = ['<lora:teary:0.1>,teary,tear,sad,sorrow,crying,open mouth']
            clothes = ['<lora:GreekClothes:0.1> , greek clothes, peplos', '<lora:LiquidClothesV1fixed:0.1>,liquid clothes', '<lora:r:0.1>,prison clothes,striped,off shoulder,striped headwear', '<lora:sd-No.414sailor_serafuku:0.1>,sailor,sailor hat']

            default_y_labels = style + face +clothes

            prompt = p.prompt
            original_prompt = prompt[0] if type(prompt) == list else prompt
            positive_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

            y_labels = default_y_labels

            combination_count = len(y_labels)

            all_prompts = []


            for i in y_labels:

                for weight1 in lora1_weight:
                    for weight2 in lora2_weight:
                        prompt = original_prompt

                        pattern1 = r'0\.2'
                        replacement1 = weight2
                        prompt = re.sub(pattern1, replacement1, prompt, count=1)

                        prompt = delimiter.join([prompt,i])

                        pattern2 = r'0\.1'
                        replacement2 = weight1
                        prompt = re.sub(pattern2, replacement2, prompt, count=1)
                        print(prompt)

                        all_prompts.append(prompt)

            p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
            p.do_not_save_grid = True

            print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

            p.prompt = all_prompts

            p.seed = [int(p.seed) for i in range(len(all_prompts))]

            p.prompt_for_display = positive_prompt

            processed = process_images(p)

            for i in range(combination_count):
                hor_texts = [[images.GridAnnotation(x)] for x in lora1_weight]
                ver_texts = [[images.GridAnnotation(y)] for y in lora2_weight]

                start_index = (i * len(lora1_weight) * len(lora2_weight)) + i
                end_index = start_index + len(lora1_weight) * len(lora2_weight)

                grid = images.image_grid(processed.images[start_index:end_index], rows=len(lora2_weight))

                grid = images.draw_grid_annotations(grid, processed.images[start_index].size[0],
                                                    processed.images[start_index].size[1], hor_texts,
                                                    ver_texts, margin_size)

                processed.images.insert(i, grid)
                processed.all_prompts.insert(i, processed.all_prompts[start_index])
                processed.all_seeds.insert(i, processed.all_seeds[start_index])
                processed.infotexts.insert(i, processed.infotexts[start_index])

                images.save_image(processed.images[i], p.outpath_grids + '/autotest/' + lora_name + '/lora_test/', ver_texts[0],
                                  extension=opts.grid_format,
                                  prompt=original_prompt, seed=processed.seed, grid=True, p=p,
                                  forced_filename=lora_name + '_' + y_labels[i])
            return processed


        #没看懂xyz_grid怎么弄的
        def control_weight_test(pc, margin_size):
            p = copy(pc)
            return

        print('=============================================================')

        if complex_checkpoint_test:

            default_y_labels = ['hand on own face', 'arms behind back, looking at viewer', 'head back, full body',
                                'armpit peek', 'hand on hip, standing, full body ', 'own hands together,  pray',
                                'outstretched arm', 'sit, crossed legs', 'hand to own mouth, close-up,',
                                'thumbs up', 'leaning forward', 'crossed arms', 'lying, from above', 'looking back',
                                'jumping, full body', 'head rest', 'reading books', 'holding umbrella', 'squat',
                                'kneeling']

            if not complex_checkpoint_values:

                if select_test_activate:
                    with SharedSettingsStackHelper():
                        processed = select_ckpt_test(p, simple_test_activate, margin_size)
                else:
                    with SharedSettingsStackHelper():
                        processed = complex_test(p, simple_test_activate, margin_size)
                with SharedSettingsStackHelper():
                    processed = complex_lora_test(p, lora_test_activate, margin_size)
            else:
                print('=============================================================')
                print(complex_checkpoint_values)

                def apply_checkpoint(p, x):
                    info = modules.sd_models.get_closet_checkpoint_match(x)
                    if info is None:
                        raise RuntimeError(f"Unknown checkpoint: {x}")
                    p.override_settings['sd_model_checkpoint'] = info.name

                for i in complex_checkpoint_values:
                    print('=============================================================')
                    print(i)
                    pp = copy(p)
                    apply_checkpoint(pp, i)
                    if select_test_activate:
                        with SharedSettingsStackHelper():
                            processed = select_ckpt_test(p, simple_test_activate, margin_size)
                    else:
                        with SharedSettingsStackHelper():
                            processed = complex_test(p, simple_test_activate, margin_size)
                    with SharedSettingsStackHelper():
                        processed = complex_lora_test(pp, lora_test_activate, margin_size)

        else:

            prompt = copy(p.prompt)
            pattern = r":(.*?):"
            match = re.search(pattern, prompt)
            lora_name = match.group(1)

            if character_test_activate:



                default_y_labels = ['walking', 'arms crossed', 'own hands together', 'arms up', 'knees to chest',
                                    'pointing at self', 'grin', 'crying, tears', '>o<']

                with SharedSettingsStackHelper():
                    processed = simple_test(p,default_y_labels, simple_test_activate, margin_size, lora_name)

                with SharedSettingsStackHelper():
                    processed = checkpoint_test(p, checkpoint_test_activate, checkpoint_values, margin_size, lora_name)

                with SharedSettingsStackHelper():
                    processed = lora_test(p, lora_test_activate, lora_values, margin_size, lora_name)
            elif style_test_activate:

                prompt = copy(p.prompt)
                pattern = r":(.*?):"
                match = re.search(pattern, prompt)
                lora_name = match.group(1)

                prompt = [
                    '<lora:genshin_impact_raiden_shogun:0.2> raiden shogun \(genshin_impact\), 1girl, solo',
                    '<lora:genshin_impact_sangonomiya_kokomi:0.2> sangonomiya kokomi \(genshin impact\) 1girl, solo',
                    '<lora:Sv4-06:0.2> SilverWolfV4, 1girl, solo',
                    '<lora:genshinImpact_scaramouche_v10:0.2>, scaramouche\(genshin impact\), (dark blue hat:1.2), dark blue hair, short hair, dark blue eyes, red eye lines, male',
                    '<lora:genshin_impact_keqing:0.2> (keqing \(genshin impact\):1.0), 1girl, solo, twintail, purple hair',
                    '<lora:Clara_Honkai_Star_Rail_v2-10:0.2> (masterpiece), best quality, 1girl, solo, clara, honkai_star_rail, white hair',
                    '<lora:genshinImpact_venti_v10:0.2> venti\(genshin_impact\), 1boy, male focus, solo, dark blue hair, green hat, green eyes, twin braids, short braids, blue braids',
                    '<lora:genshin_impact_eula:0.2> eula \(genshin impact\), 1girl, solo, bangs, hairband',
                    '<lora:bocchitherock_gotouhitori_v1:0.2>, gotou1, gotou hitori, 1girl, solo, pink hair, long hair, hair ornament, blue eyes,',
                    '<lora:genshinimpact_nilou:0.2>, (niloudef:0.8), 1girl, solo, orange hair, long hair, twin tails, green eyes,'
                ]

                name = ['raiden_shogun', 'kokomi', 'silver_wolf', 'scaramouche', 'keqing', 'clara', 'venti', 'eula',
                        'gotouhitori', 'nilou']

                #for p_prompt, lora_name in zip(prompt, name):
                for p_prompt in prompt:
                    pp = copy(p)

                    pp.prompt = p.prompt + p_prompt

                    default_y_labels = ['half body']

                    with SharedSettingsStackHelper():
                        processed = simple_test(pp, default_y_labels, simple_test_activate, margin_size, lora_name)

                    with SharedSettingsStackHelper():
                        processed = checkpoint_test(pp, checkpoint_test_activate, checkpoint_values, margin_size,
                                                    lora_name)

                    with SharedSettingsStackHelper():
                        processed = lora_test(pp, lora_test_activate, lora_values, margin_size, lora_name)
            else:
                default_y_labels = ['hand on own face', 'arms behind back, looking at viewer', 'head back, full body',
                                    'armpit peek', 'hand on hip, standing, full body ', 'own hands together,  pray',
                                    'outstretched arm', 'sit, crossed legs', 'hand to own mouth, close-up,',
                                    'thumbs up', 'leaning forward', 'crossed arms', 'lying, from above', 'looking back',
                                    'jumping, full body', 'head rest', 'reading books', 'holding umbrella', 'squat',
                                    'kneeling']

                with SharedSettingsStackHelper():
                    processed = simple_test(p, default_y_labels, simple_test_activate, margin_size, lora_name)

                with SharedSettingsStackHelper():
                    processed = checkpoint_test(p, checkpoint_test_activate, checkpoint_values, margin_size, lora_name)

                with SharedSettingsStackHelper():
                    processed = lora_test(p, lora_test_activate, lora_values, margin_size, lora_name)


        #print("1 simple_test successed")
        #with SharedSettingsStackHelper():
        #    processed = simple_test(p, margin_size)

        #grid = images.image_grid(processed.images, p.batch_size, rows=1 << ((len(prompt_matrix_parts) - 1) // 2))
        #grid = images.draw_prompt_matrix(grid, processed.images[0].width, processed.images[0].height, prompt_matrix_parts, margin_size)
        return processed





if __name__ == "__main__":

    output_path = 'test_output/'

    if not os.path.exists(output_path):
        # 创建文件夹
        os.mkdir(output_path)
    else:
        print("文件夹已经存在")
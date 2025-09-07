import os
import logging
from typing import List
from PIL import Image

class MLService:
    def __init__(self):
        self.enabled = True

        # Feature flags
        self.hf_enabled = False  # Hugging Face captioning (BLIP or ViT-GPT2)
        self.caption_model_type = None  # 'blip' | 'vitgpt2'
        self.gcv_enabled = False  # Google Cloud Vision (preferred when available)

        # Try Google Cloud Vision first (no hardcoded templates when available)
        try:
            # Lazy import so environments without the lib still work
            from google.cloud import vision  # type: ignore
            from google.oauth2 import service_account  # type: ignore

            creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not creds_path:
                # Fallback to repo credential if present
                candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'google-credentials.json')
                if os.path.exists(candidate):
                    creds_path = candidate

            if creds_path and os.path.exists(creds_path):
                credentials = service_account.Credentials.from_service_account_file(creds_path)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                self.gcv_enabled = True
                logging.info('Google Cloud Vision enabled for MLService')
            else:
                logging.info('Google Cloud Vision credentials not found; skipping GCV integration')
        except Exception as e:
            logging.info(f'Google Cloud Vision not available: {e}')

        # Try to initialize Hugging Face BLIP (optional, only if weights are cached locally)
        if not self.gcv_enabled:
            # Try BLIP first
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.hf_enabled = True
                self.caption_model_type = 'blip'
                logging.info("Hugging Face BLIP model loaded successfully")
            except Exception as e:
                logging.info(f"BLIP not available: {e}")
                # Try a smaller alternative: ViT-GPT2
                try:
                    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer  # type: ignore
                    self.vitgpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                    self.vitgpt2_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                    self.vitgpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                    self.hf_enabled = True
                    self.caption_model_type = 'vitgpt2'
                    logging.info("Hugging Face ViT-GPT2 captioning model loaded successfully")
                except Exception as e2:
                    logging.info(f"Hugging Face models not available, will use basic analysis: {e2}")
    
    # Removed hardcoded/randomized prose templates. We now base outputs on
    # actual detections (GCV/BLIP) and deterministic fallbacks only.

    def _generate_fallback_description(self, image_path: str) -> str:
        """Deterministic fallback: describe by filename and orientation."""
        name = os.path.splitext(os.path.basename(image_path))[0].replace('_', ' ').replace('-', ' ').strip()
        try:
            img = Image.open(image_path)
            w, h = img.size
        except Exception:
            w = h = 0
        orient = 'square' if w == h else ('wide' if w > h else 'tall') if w and h else 'photo'
        if name:
            return f"Photo of {name} ({orient})."
        return f"Photo ({orient})."
    
    def generate_alt_text(self, image_path):
        """Generate alt text leaning on real model outputs when available."""
        if not self.enabled:
            return self._generate_fallback_description(image_path)

        # Prefer Google Cloud Vision labels/objects
        if self.gcv_enabled:
            try:
                return self._generate_gcv_alt_text(image_path)
            except Exception as e:
                logging.warning(f"GCV alt text generation failed: {e}")

        # Try Hugging Face BLIP
        if self.hf_enabled:
            try:
                return self._generate_hf_alt_text(image_path)
            except Exception as e:
                logging.warning(f"Hugging Face alt text generation failed: {e}")

        # Final deterministic fallback
        return self._generate_fallback_description(image_path)
    
    def _generate_hf_alt_text(self, image_path):
        """Generate alt text using Hugging Face BLIP model"""
        try:
            from PIL import Image as PILImage
            
            # Load and process the image
            image = PILImage.open(image_path).convert('RGB')
            
            if self.caption_model_type == 'blip':
                # Generate caption using BLIP
                inputs = self.processor(image, return_tensors="pt")
                out = self.model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            elif self.caption_model_type == 'vitgpt2':
                # Generate caption using ViT-GPT2
                pixel_values = self.vitgpt2_processor(images=image, return_tensors="pt").pixel_values
                output_ids = self.vitgpt2_model.generate(pixel_values, max_length=50, num_beams=5)
                caption = self.vitgpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                raise RuntimeError("HF model type not initialized")
            
            # Clean up and enhance the caption
            if caption:
                # Capitalize first letter and ensure it ends properly
                caption = caption.strip()
                if caption and not caption[0].isupper():
                    caption = caption[0].upper() + caption[1:]
                if caption and not caption.endswith('.'):
                    caption += '.'
                return caption
            else:
                return self._generate_basic_alt_text(image_path)
                
        except Exception as e:
            logging.error(f"HuggingFace BLIP generation failed: {e}")
            raise e
    
    def _generate_basic_alt_text(self, image_path):
        """Deterministic, minimal alt text using filename and orientation only."""
        try:
            return self._generate_fallback_description(image_path)
        except Exception as e:
            logging.error(f"Basic alt text generation failed: {e}")
            return "Photo."
    
    def detect_objects(self, image_path):
        """Detect objects/labels for search, preferring real model outputs."""
        if not self.enabled:
            return []
        
        try:
            # Prefer Google Cloud Vision when available
            if self.gcv_enabled:
                return self._detect_objects_gcv(image_path)

            # Try Hugging Face next if available
            if self.hf_enabled:
                return self._detect_objects_hf(image_path)
            
            # Fallback to basic PIL analysis
            return self._detect_objects_basic(image_path)
        except Exception as e:
            logging.error(f"Object detection failed: {e}")
            return []
    
    def _detect_objects_hf(self, image_path):
        """Detect objects using HuggingFace model"""
        try:
            # Use BLIP caption to extract keywords deterministically
            alt_text = self._generate_hf_alt_text(image_path)
            tokens = [t.strip('.,;:!?()').lower() for t in alt_text.split()]
            # Return nouns-like tokens heuristically (keep a simple allowlist)
            allow = {
                'person','people','man','woman','child','baby','boy','girl',
                'dog','cat','bird','animal','horse','cow','sheep','bear','zebra','giraffe',
                'car','truck','bike','bicycle','motorcycle','bus','vehicle',
                'house','building','tree','flower','plant','grass','sky','cloud',
                'food','cake','pizza','sandwich','fruit','table','chair','bed','sofa',
                'water','ocean','lake','river','beach','mountain','sun','computer','laptop','phone','screen'
            }
            detected = [t for t in tokens if t in allow]
            # Deduplicate preserving order
            seen = set()
            result = []
            for t in detected:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
            return result[:10]
        except Exception as e:
            logging.error(f"HF object detection failed: {e}")
            return self._detect_objects_basic(image_path)
    
    def _detect_objects_basic(self, image_path):
        """Basic object detection using filename and image properties"""
        try:
            detected_objects = []
            # Infer objects from filename only (deterministic, minimal)
            for token in os.path.splitext(os.path.basename(image_path))[0].lower().replace('-', ' ').replace('_', ' ').split():
                if token.isalpha() and len(token) > 2:
                    detected_objects.append(token)
            # De-duplicate while preserving order
            seen = set()
            out = []
            for t in detected_objects:
                if t not in seen:
                    out.append(t)
                    seen.add(t)
            return out[:10]
        
        except Exception as e:
            logging.error(f"Basic object detection failed: {e}")
            return []

    # ----------------------------
    # Google Cloud Vision helpers
    # ----------------------------
    def _generate_gcv_alt_text(self, image_path: str) -> str:
        from google.cloud import vision  # type: ignore

        with open(image_path, 'rb') as f:
            content = f.read()

        image = vision.Image(content=content)

        # Fetch labels and localized objects
        labels_resp = self.vision_client.label_detection(image=image)
        objects_resp = self.vision_client.object_localization(image=image)
        props_resp = self.vision_client.image_properties(image=image)

        labels = [l.description for l in (labels_resp.label_annotations or [])]
        objects = [o.name for o in (objects_resp.localized_object_annotations or [])]

        # Choose a primary subject: prefer object over label for specificity
        primary = None
        preferred = ['dog','cat','person','car','flower','tree','bird']
        lower_objects = [o.lower() for o in objects]
        lower_labels = [l.lower() for l in labels]
        for p in preferred:
            if p in lower_objects:
                primary = p
                break
        if not primary:
            for p in preferred:
                if p in lower_labels:
                    primary = p
                    break
        if not primary and lower_objects:
            primary = lower_objects[0]
        if not primary and lower_labels:
            primary = lower_labels[0]

        # Add context keyword if available
        context = None
        context_candidates = ['outdoor','indoor','grass','park','street','beach','mountain','sky']
        for c in context_candidates:
            if c in lower_labels or c in lower_objects:
                context = c
                break

        # Consider general colorfulness to optionally add a qualifier (no prose)
        qualifier = None
        try:
            dom_colors = (props_resp.image_properties_annotation.dominant_colors.colors or [])
            if dom_colors:
                # Use the most prominent color
                top = max(dom_colors, key=lambda c: getattr(c, 'pixel_fraction', 0.0))
                r = int(top.color.red or 0)
                g = int(top.color.green or 0)
                b = int(top.color.blue or 0)
                # Classify simple color bucket
                if r > g + 30 and r > b + 30:
                    qualifier = 'red-toned'
                elif g > r + 30 and g > b + 30:
                    qualifier = 'green-toned'
                elif b > r + 30 and b > g + 30:
                    qualifier = 'blue-toned'
        except Exception:
            pass

        parts = []
        if primary:
            parts.append(primary)
        if context and context != primary:
            parts.append(f"in {context}")
        if qualifier:
            parts.append(f"({qualifier})")

        if parts:
            # Capitalize first word
            sentence = ' '.join(parts).strip()
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith('.'):
                sentence += '.'
            return sentence

        # If nothing useful, fallback deterministically
        return self._generate_fallback_description(image_path)

    def _detect_objects_gcv(self, image_path: str) -> List[str]:
        from google.cloud import vision  # type: ignore

        with open(image_path, 'rb') as f:
            content = f.read()
        image = vision.Image(content=content)

        labels_resp = self.vision_client.label_detection(image=image)
        objects_resp = self.vision_client.object_localization(image=image)

        labels = [l.description.lower() for l in (labels_resp.label_annotations or [])]
        objects = [o.name.lower() for o in (objects_resp.localized_object_annotations or [])]

        # Merge, de-duplicate, prefer objects first
        seen = set()
        out: List[str] = []
        for t in objects + labels:
            if t not in seen:
                out.append(t)
                seen.add(t)
            if len(out) >= 10:
                break
        return out

# Global instance
ml_service = MLService()

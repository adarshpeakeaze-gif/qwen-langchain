"""
Document Assistant for Blind Users
With LangChain integration for document classification and structured extraction

A slim Flask application that uses modular components:
- templates/index.html: Frontend UI
- prompts/: Document processing prompts
- extraction/: LangChain tools, classifier agent, and key rotation
- config.py: Centralized configuration
"""

import base64
import io
import logging
import uuid
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from config import config


def generate_request_id() -> str:
    """Generate unique request ID for tracking."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique = uuid.uuid4().hex[:8]
    return f"doc_{timestamp}_{unique}"


from prompts import BLIND_DESCRIPTION_PROMPT
from extraction import classify_document, perform_structured_extraction
from extraction.classifier import call_openrouter_vision, get_key_status

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH


def convert_pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of PNG image bytes (one per page)."""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    max_pages = min(len(pdf_document), config.PDF_MAX_PAGES)
    for page_num in range(max_pages):
        page = pdf_document[page_num]
        mat = fitz.Matrix(config.PDF_DPI_SCALE, config.PDF_DPI_SCALE)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)

    pdf_document.close()
    return images


def optimize_image_for_vision(image_bytes: bytes, max_dimension: int = None, quality: int = None) -> tuple:
    """
    Optimize image for vision API to reduce processing time.

    - Resizes large images to max_dimension while maintaining aspect ratio
    - Converts to JPEG for smaller file size (unless transparency is needed)
    - Returns (optimized_bytes, mime_type, original_size, new_size)

    Max dimension of 1568 is optimal for most vision models (Qwen, GPT-4V, Claude)
    as they typically process at ~1500px internally anyway.
    """
    # Use config defaults if not specified
    if max_dimension is None:
        max_dimension = config.IMAGE_MAX_DIMENSION
    if quality is None:
        quality = config.IMAGE_QUALITY

    original_size = len(image_bytes)

    # Skip optimization if disabled
    if not config.IMAGE_OPTIMIZATION_ENABLED:
        # Determine mime type from image
        img = Image.open(io.BytesIO(image_bytes))
        mime_type = 'image/png' if img.format == 'PNG' else 'image/jpeg'
        return image_bytes, mime_type, original_size, original_size

    # Open image with PIL
    img = Image.open(io.BytesIO(image_bytes))

    # Convert RGBA to RGB if no transparency (for JPEG conversion)
    has_transparency = img.mode in ('RGBA', 'LA', 'P')
    if has_transparency and img.mode == 'P':
        img = img.convert('RGBA')
        has_transparency = 'A' in img.getbands()

    # Get original dimensions
    orig_width, orig_height = img.size

    # Calculate new dimensions maintaining aspect ratio
    if max(orig_width, orig_height) > max_dimension:
        if orig_width > orig_height:
            new_width = max_dimension
            new_height = int(orig_height * (max_dimension / orig_width))
        else:
            new_height = max_dimension
            new_width = int(orig_width * (max_dimension / orig_height))

        # Use high-quality downsampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to RGB if no transparency needed (for JPEG)
    if not has_transparency:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPEG
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)
        optimized_bytes = output.read()
        mime_type = 'image/jpeg'
    else:
        # Keep as PNG for transparent images
        output = io.BytesIO()
        img.save(output, format='PNG', optimize=True)
        output.seek(0)
        optimized_bytes = output.read()
        mime_type = 'image/png'

    new_size = len(optimized_bytes)

    return optimized_bytes, mime_type, original_size, new_size


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Get API configuration and key rotation status."""
    key_status = get_key_status()
    return jsonify({
        'configured': bool(config.OPENROUTER_API_KEYS),
        'vision_model': config.VISION_MODEL,
        'classifier_model': config.CLASSIFIER_MODEL,
        'extraction_model': config.EXTRACTION_MODEL,
        'key_status': key_status,
        'config': {
            'max_retries': config.MAX_RETRIES,
            'max_concurrent': config.MAX_CONCURRENT_REQUESTS,
            'vision_max_tokens': config.VISION_MAX_TOKENS,
            'classifier_max_tokens': config.CLASSIFIER_MAX_TOKENS,
            'extraction_max_tokens': config.EXTRACTION_MAX_TOKENS,
        }
    })


@app.route('/api/keys/status')
def keys_status():
    """Get detailed key rotation status."""
    return jsonify(get_key_status())


@app.route('/api/describe', methods=['POST'])
def describe():
    """
    Describe a document image using vision model + classify with LangChain.
    Uses key rotation and retry logic for reliability.
    """
    request_id = generate_request_id()
    start_time = datetime.now()

    logger.info(f"[{request_id}] === NEW REQUEST: /api/describe ===")

    if 'file' not in request.files:
        logger.warning(f"[{request_id}] No file provided in request")
        return jsonify({'success': False, 'error': 'No file provided', 'request_id': request_id})

    file = request.files['file']
    if file.filename == '':
        logger.warning(f"[{request_id}] Empty filename")
        return jsonify({'success': False, 'error': 'No file selected', 'request_id': request_id})

    logger.info(f"[{request_id}] Processing file: {file.filename}")

    try:
        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        filename = secure_filename(file.filename).lower()
        logger.info(f"[{request_id}] File size: {file_size} bytes | filename: {filename}")

        # Check if PDF and convert to images
        if filename.endswith('.pdf'):
            logger.info(f"[{request_id}] Converting PDF to images...")
            pdf_images = convert_pdf_to_images(file_content)
            if not pdf_images:
                logger.error(f"[{request_id}] Failed to extract images from PDF")
                return jsonify({'success': False, 'error': 'Could not extract images from PDF', 'request_id': request_id})

            image_bytes = pdf_images[0]
            num_pages = len(pdf_images)
            logger.info(f"[{request_id}] PDF converted | pages={num_pages} | using page 1")
        else:
            image_bytes = file_content
            num_pages = 1
            logger.info(f"[{request_id}] Image file | size={len(image_bytes)} bytes")

        # Optimize image for vision API (resize and compress)
        logger.info(f"[{request_id}] Optimizing image for vision API...")
        optimized_bytes, mime_type, orig_size, new_size = optimize_image_for_vision(image_bytes)
        compression_ratio = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
        logger.info(f"[{request_id}] Image optimized | {orig_size/1024:.1f}KB -> {new_size/1024:.1f}KB | {compression_ratio:.1f}% reduction")

        # Encode optimized image bytes to base64
        image_base64 = base64.b64encode(optimized_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{image_base64}"
        logger.debug(f"[{request_id}] Base64 encoded | length={len(image_base64)} chars")

        # Modify prompt for multi-page PDFs
        prompt = BLIND_DESCRIPTION_PROMPT
        if num_pages > 1:
            prompt += f"\n\nNote: This is a {num_pages}-page PDF document. You are viewing page 1 of {num_pages}."

        # Step 1: Call Vision LLM (with key rotation)
        logger.info(f"[{request_id}] Step 1: Starting vision analysis...")
        vision_start = datetime.now()

        description = call_openrouter_vision(
            image_url=image_url,
            prompt=prompt,
            max_tokens=config.VISION_MAX_TOKENS,
            temperature=config.VISION_TEMPERATURE,
            timeout=config.VISION_TIMEOUT,
            request_id=request_id,
        )

        vision_time = (datetime.now() - vision_start).total_seconds()
        logger.info(f"[{request_id}] Step 1 completed | time={vision_time:.2f}s | description_length={len(description)} chars")

        # Step 2: Pass description to LangChain classifier agent
        logger.info(f"[{request_id}] Step 2: Starting classification...")
        classify_start = datetime.now()

        classification_result = classify_document(description, request_id=request_id)

        classify_time = (datetime.now() - classify_start).total_seconds()
        doc_type = classification_result.get('document_type', 'unknown')
        logger.info(f"[{request_id}] Step 2 completed | time={classify_time:.2f}s | document_type={doc_type}")

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{request_id}] === REQUEST COMPLETE === | total_time={processing_time:.2f}s | file={file.filename} | type={doc_type}")

        return jsonify({
            'success': True,
            'request_id': request_id,
            'filename': file.filename,
            'processing_time_seconds': processing_time,
            'timing': {
                'vision_seconds': vision_time,
                'classification_seconds': classify_time,
            },
            'description': description,
            'classification': classification_result
        })

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{request_id}] === REQUEST FAILED === | time={processing_time:.2f}s | error={e}", exc_info=True)
        return jsonify({
            'success': False,
            'request_id': request_id,
            'error': str(e)
        })


@app.route('/api/extract', methods=['POST'])
def extract_structured():
    """
    Endpoint for structured data extraction from a document.
    Expects a file upload and optionally a document_type hint.
    Uses key rotation and retry logic.
    """
    request_id = generate_request_id()
    start_time = datetime.now()

    logger.info(f"[{request_id}] === NEW REQUEST: /api/extract ===")

    if 'file' not in request.files:
        logger.warning(f"[{request_id}] No file provided in request")
        return jsonify({'success': False, 'error': 'No file provided', 'request_id': request_id})

    file = request.files['file']
    if file.filename == '':
        logger.warning(f"[{request_id}] Empty filename")
        return jsonify({'success': False, 'error': 'No file selected', 'request_id': request_id})

    document_type = request.form.get('document_type', None)
    logger.info(f"[{request_id}] Processing file: {file.filename} | doc_type_hint={document_type}")

    try:
        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        filename = secure_filename(file.filename).lower()
        logger.info(f"[{request_id}] File size: {file_size} bytes | filename: {filename}")

        # Check if PDF and convert to images
        if filename.endswith('.pdf'):
            logger.info(f"[{request_id}] Converting PDF to images...")
            pdf_images = convert_pdf_to_images(file_content)
            if not pdf_images:
                logger.error(f"[{request_id}] Failed to extract images from PDF")
                return jsonify({'success': False, 'error': 'Could not extract images from PDF', 'request_id': request_id})
            image_bytes = pdf_images[0]
            logger.info(f"[{request_id}] PDF converted | pages={len(pdf_images)}")
        else:
            image_bytes = file_content
            logger.info(f"[{request_id}] Image file | size={len(image_bytes)} bytes")

        # Optimize image for vision API (resize and compress)
        logger.info(f"[{request_id}] Optimizing image for vision API...")
        optimized_bytes, mime_type, orig_size, new_size = optimize_image_for_vision(image_bytes)
        compression_ratio = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
        logger.info(f"[{request_id}] Image optimized | {orig_size/1024:.1f}KB -> {new_size/1024:.1f}KB | {compression_ratio:.1f}% reduction")

        # Encode optimized image bytes to base64
        image_base64 = base64.b64encode(optimized_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{image_base64}"

        # Perform structured extraction (with key rotation)
        logger.info(f"[{request_id}] Starting structured extraction...")
        extraction_result = perform_structured_extraction(image_url, document_type, request_id=request_id)

        processing_time = (datetime.now() - start_time).total_seconds()

        extraction_result['request_id'] = request_id
        extraction_result['filename'] = file.filename
        extraction_result['processing_time_seconds'] = processing_time

        logger.info(f"[{request_id}] === REQUEST COMPLETE === | total_time={processing_time:.2f}s | success={extraction_result.get('success')}")

        return jsonify(extraction_result)

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{request_id}] === REQUEST FAILED === | time={processing_time:.2f}s | error={e}", exc_info=True)
        return jsonify({
            'success': False,
            'request_id': request_id,
            'error': str(e)
        })


@app.route('/api/process', methods=['POST'])
def process_full():
    """
    Full pipeline: describe + classify + extract structured data.
    """
    request_id = generate_request_id()
    start_time = datetime.now()

    logger.info(f"[{request_id}] === NEW REQUEST: /api/process (FULL PIPELINE) ===")

    if 'file' not in request.files:
        logger.warning(f"[{request_id}] No file provided in request")
        return jsonify({'success': False, 'error': 'No file provided', 'request_id': request_id})

    file = request.files['file']
    if file.filename == '':
        logger.warning(f"[{request_id}] Empty filename")
        return jsonify({'success': False, 'error': 'No file selected', 'request_id': request_id})

    logger.info(f"[{request_id}] Processing file: {file.filename}")

    try:
        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        filename = secure_filename(file.filename).lower()
        logger.info(f"[{request_id}] File size: {file_size} bytes | filename: {filename}")

        # Check if PDF and convert to images
        if filename.endswith('.pdf'):
            logger.info(f"[{request_id}] Converting PDF to images...")
            pdf_images = convert_pdf_to_images(file_content)
            if not pdf_images:
                logger.error(f"[{request_id}] Failed to extract images from PDF")
                return jsonify({'success': False, 'error': 'Could not extract images from PDF', 'request_id': request_id})
            image_bytes = pdf_images[0]
            num_pages = len(pdf_images)
            logger.info(f"[{request_id}] PDF converted | pages={num_pages} | using page 1")
        else:
            image_bytes = file_content
            num_pages = 1
            logger.info(f"[{request_id}] Image file | size={len(image_bytes)} bytes")

        # Optimize image for vision API (resize and compress)
        logger.info(f"[{request_id}] Optimizing image for vision API...")
        optimized_bytes, mime_type, orig_size, new_size = optimize_image_for_vision(image_bytes)
        compression_ratio = (1 - new_size / orig_size) * 100 if orig_size > 0 else 0
        logger.info(f"[{request_id}] Image optimized | {orig_size/1024:.1f}KB -> {new_size/1024:.1f}KB | {compression_ratio:.1f}% reduction")

        # Encode optimized image bytes to base64
        image_base64 = base64.b64encode(optimized_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{image_base64}"

        # Modify prompt for multi-page PDFs
        prompt = BLIND_DESCRIPTION_PROMPT
        if num_pages > 1:
            prompt += f"\n\nNote: This is a {num_pages}-page PDF document. You are viewing page 1 of {num_pages}."

        # Step 1: Vision description
        logger.info(f"[{request_id}] Step 1/3: Starting vision analysis...")
        vision_start = datetime.now()

        description = call_openrouter_vision(
            image_url=image_url,
            prompt=prompt,
            max_tokens=config.VISION_MAX_TOKENS,
            temperature=config.VISION_TEMPERATURE,
            timeout=config.VISION_TIMEOUT,
            request_id=request_id,
        )

        vision_time = (datetime.now() - vision_start).total_seconds()
        logger.info(f"[{request_id}] Step 1/3 completed | time={vision_time:.2f}s")

        # Step 2: Classification
        logger.info(f"[{request_id}] Step 2/3: Starting classification...")
        classify_start = datetime.now()

        classification_result = classify_document(description, request_id=request_id)
        doc_type = classification_result.get('document_type', 'unknown')

        classify_time = (datetime.now() - classify_start).total_seconds()
        logger.info(f"[{request_id}] Step 2/3 completed | time={classify_time:.2f}s | document_type={doc_type}")

        # Step 3: Structured extraction
        logger.info(f"[{request_id}] Step 3/3: Starting structured extraction...")
        extract_start = datetime.now()

        extraction_result = perform_structured_extraction(image_url, doc_type, request_id=request_id)

        extract_time = (datetime.now() - extract_start).total_seconds()
        logger.info(f"[{request_id}] Step 3/3 completed | time={extract_time:.2f}s | success={extraction_result.get('success')}")

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[{request_id}] === REQUEST COMPLETE === | total_time={processing_time:.2f}s | file={file.filename} | type={doc_type}")

        return jsonify({
            'success': True,
            'request_id': request_id,
            'filename': file.filename,
            'processing_time_seconds': processing_time,
            'timing': {
                'vision_seconds': vision_time,
                'classification_seconds': classify_time,
                'extraction_seconds': extract_time,
            },
            'description': description,
            'classification': classification_result,
            'extraction': extraction_result
        })

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{request_id}] === REQUEST FAILED === | time={processing_time:.2f}s | error={e}", exc_info=True)
        return jsonify({
            'success': False,
            'request_id': request_id,
            'error': str(e)
        })


if __name__ == '__main__':
    # Print configuration summary
    config.print_config()

    # Validate configuration
    issues = config.validate()
    for issue in issues:
        print(issue)

    print(f"\nStarting server at http://{config.HOST}:{config.PORT}")
    print("Press Ctrl+C to stop\n")

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )

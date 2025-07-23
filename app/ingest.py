import os
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IngestSettings(BaseSettings):
    """Configuration for document ingestion."""
    data_dir: str = os.path.join(os.path.dirname(__file__), '../data')
    index_dir: str = os.path.join(os.path.dirname(__file__), '../faiss_index')
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 100  # Process embeddings in batches
    supported_extensions: List[str] = ['.pdf']
    
    class Config:
        env_prefix = "INGEST_"

settings = IngestSettings()

class DocumentIngestor:
    """Enhanced document ingestion with progress tracking and error handling."""
    
    def __init__(self, settings: IngestSettings):
        self.settings = settings
        self.embeddings = None
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_documents': 0,
            'total_chunks': 0,
            'processing_time': 0,
            'failed_files_list': []
        }
    
    def _initialize_embeddings(self):
        """Initialize the embeddings model."""
        if self.embeddings is None:
            try:
                logger.info(f"Loading embeddings model: {self.settings.embedding_model}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embeddings model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {str(e)}")
                raise
    
    def _get_supported_files(self) -> List[Path]:
        """Get list of supported files from data directory."""
        data_path = Path(self.settings.data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.settings.data_dir}")
        
        supported_files = []
        for ext in self.settings.supported_extensions:
            supported_files.extend(data_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(supported_files)} supported files")
        return supported_files
    
    def _load_document(self, file_path: Path) -> List[Document]:
        """Load a single document with error handling."""
        try:
            logger.info(f"Loading document: {file_path.name}")
            
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update({
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_size': file_path.stat().st_size,
                        'ingestion_date': datetime.now().isoformat(),
                        'file_extension': file_path.suffix.lower()
                    })
                
                logger.info(f"Successfully loaded {len(docs)} pages from {file_path.name}")
                return docs
                
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {str(e)}")
            self.stats['failed_files_list'].append({
                'file': str(file_path),
                'error': str(e)
            })
            return []
    
    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks with enhanced metadata."""
        try:
            logger.info(f"Splitting {len(docs)} documents into chunks...")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_documents(docs)
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'total_chunks': len(chunks)
                })
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split documents: {str(e)}")
            raise
    
    def _create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vectorstore with batch processing."""
        try:
            logger.info("Creating embeddings and FAISS vectorstore...")
            self._initialize_embeddings()
            
            # Process in batches for better memory management
            batch_size = self.settings.batch_size
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            vectorstore = None
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                if vectorstore is None:
                    # Create initial vectorstore
                    vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    # Add to existing vectorstore
                    batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                    vectorstore.merge_from(batch_vectorstore)
            
            logger.info("Vectorstore created successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {str(e)}")
            raise
    
    def _save_vectorstore(self, vectorstore: FAISS):
        """Save vectorstore with backup and metadata."""
        try:
            index_path = Path(self.settings.index_dir)
            
            # Create backup if index already exists
            if index_path.exists():
                backup_path = index_path.parent / f"{index_path.name}_backup_{int(time.time())}"
                logger.info(f"Creating backup at: {backup_path}")
                import shutil
                shutil.move(str(index_path), str(backup_path))
            
            # Create directory if it doesn't exist
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save vectorstore
            logger.info(f"Saving vectorstore to: {index_path}")
            vectorstore.save_local(str(index_path))
            
            # Save ingestion metadata
            metadata = {
                'ingestion_date': datetime.now().isoformat(),
                'settings': {
                    'embedding_model': self.settings.embedding_model,
                    'chunk_size': self.settings.chunk_size,
                    'chunk_overlap': self.settings.chunk_overlap,
                    'batch_size': self.settings.batch_size
                },
                'stats': self.stats,
                'total_vectors': vectorstore.index.ntotal,
                'embedding_dimension': vectorstore.index.d
            }
            
            metadata_path = index_path / 'ingestion_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Vectorstore and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save vectorstore: {str(e)}")
            raise
    
    def ingest(self) -> Dict[str, Any]:
        """Main ingestion process with comprehensive error handling."""
        start_time = time.time()
        
        try:
            logger.info("=== Starting Document Ingestion ===")
            logger.info(f"Data directory: {self.settings.data_dir}")
            logger.info(f"Index directory: {self.settings.index_dir}")
            logger.info(f"Chunk size: {self.settings.chunk_size}")
            logger.info(f"Chunk overlap: {self.settings.chunk_overlap}")
            
            # Get supported files
            files = self._get_supported_files()
            if not files:
                raise ValueError("No supported files found in data directory")
            
            self.stats['total_files'] = len(files)
            
            # Load all documents
            all_docs = []
            for file_path in files:
                docs = self._load_document(file_path)
                if docs:
                    all_docs.extend(docs)
                    self.stats['processed_files'] += 1
                else:
                    self.stats['failed_files'] += 1
            
            if not all_docs:
                raise ValueError("No documents were successfully loaded")
            
            self.stats['total_documents'] = len(all_docs)
            logger.info(f"Successfully loaded {len(all_docs)} documents from {self.stats['processed_files']} files")
            
            # Split documents
            chunks = self._split_documents(all_docs)
            self.stats['total_chunks'] = len(chunks)
            
            # Create and save vectorstore
            vectorstore = self._create_vectorstore(chunks)
            self._save_vectorstore(vectorstore)
            
            # Calculate final stats
            self.stats['processing_time'] = time.time() - start_time
            
            logger.info("=== Ingestion Completed Successfully ===")
            logger.info(f"Files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
            logger.info(f"Documents loaded: {self.stats['total_documents']}")
            logger.info(f"Chunks created: {self.stats['total_chunks']}")
            logger.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
            
            if self.stats['failed_files'] > 0:
                logger.warning(f"Failed to process {self.stats['failed_files']} files")
                for failed in self.stats['failed_files_list']:
                    logger.warning(f"  - {failed['file']}: {failed['error']}")
            
            return {
                'success': True,
                'stats': self.stats,
                'message': 'Ingestion completed successfully'
            }
            
        except Exception as e:
            self.stats['processing_time'] = time.time() - start_time
            logger.error(f"Ingestion failed: {str(e)}")
            return {
                'success': False,
                'stats': self.stats,
                'error': str(e),
                'message': 'Ingestion failed'
            }

def validate_environment() -> bool:
    """Validate that the environment is ready for ingestion."""
    try:
        # Check data directory
        data_path = Path(settings.data_dir)
        if not data_path.exists():
            logger.error(f"Data directory does not exist: {settings.data_dir}")
            return False
        
        # Check for supported files
        supported_files = []
        for ext in settings.supported_extensions:
            supported_files.extend(data_path.glob(f"*{ext}"))
        
        if not supported_files:
            logger.error(f"No supported files found in {settings.data_dir}")
            logger.info(f"Supported extensions: {settings.supported_extensions}")
            return False
        
        # Check write permissions for index directory
        index_path = Path(settings.index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        test_file = index_path / 'test_write.tmp'
        try:
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            logger.error(f"Cannot write to index directory: {e}")
            return False
        
        logger.info("Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        return False

def ingest(force: bool = False) -> Dict[str, Any]:
    """Main ingest function with validation."""
    try:
        # Validate environment
        if not validate_environment():
            return {
                'success': False,
                'error': 'Environment validation failed',
                'message': 'Please check the logs for details'
            }
        
        # Check if index already exists
        index_path = Path(settings.index_dir)
        if index_path.exists() and not force:
            logger.warning("Index already exists. Use force=True to overwrite.")
            return {
                'success': False,
                'error': 'Index already exists',
                'message': 'Use force=True to overwrite existing index'
            }
        
        # Perform ingestion
        ingestor = DocumentIngestor(settings)
        return ingestor.ingest()
        
    except Exception as e:
        logger.error(f"Ingest function failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Ingestion process failed'
        }

def get_ingestion_info() -> Dict[str, Any]:
    """Get information about the current ingestion."""
    try:
        index_path = Path(settings.index_dir)
        metadata_path = index_path / 'ingestion_metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {'error': 'No ingestion metadata found'}
            
    except Exception as e:
        return {'error': f'Failed to read ingestion info: {str(e)}'}

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest documents into FAISS vector store')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing index')
    parser.add_argument('--info', action='store_true', help='Show ingestion information')
    parser.add_argument('--validate', action='store_true', help='Validate environment only')
    
    args = parser.parse_args()
    
    if args.info:
        info = get_ingestion_info()
        print(json.dumps(info, indent=2))
    elif args.validate:
        valid = validate_environment()
        print(f"Environment valid: {valid}")
    else:
        result = ingest(force=args.force)
        print(json.dumps(result, indent=2))
        
        if not result['success']:
            exit(1)
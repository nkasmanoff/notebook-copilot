import os
import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional


class JupyterFIMGenerator:
    """
    Generate Fill-in-the-Middle (FIM) examples from Jupyter Notebooks (.ipynb files).
    This handles the unique structure of notebooks, focusing on code cells while
    maintaining the context of the entire notebook.
    """
    
    def __init__(self, min_cell_length: int = 10, min_code_cells: int = 2):
        """
        Initialize the generator with configuration parameters.
        
        Args:
            min_cell_length: Minimum character length of a code cell to be considered
            min_code_cells: Minimum number of code cells required in a notebook
        """
        self.min_cell_length = min_cell_length
        self.min_code_cells = min_code_cells
    
    def load_notebook(self, notebook_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a Jupyter notebook from file.
        
        Args:
            notebook_path: Path to the .ipynb file
            
        Returns:
            Parsed notebook content or None if loading fails
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading notebook {notebook_path}: {e}")
            return None
    
    def extract_code_cells(self, notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all code cells from a Jupyter notebook.
        
        Args:
            notebook: Parsed notebook content
            
        Returns:
            List of code cells
        """
        code_cells = []
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') in ['code', 'markdown']:
                # Only include cells with sufficient content
                source = self._get_cell_source(cell)
                if len(source) >= self.min_cell_length:
                    code_cells.append(cell)
                    
        return code_cells
    
    def _get_cell_source(self, cell: Dict[str, Any]) -> str:
        """
        Extract the source code from a cell, handling both string and list formats.
        
        Args:
            cell: Cell object from notebook
            
        Returns:
            Source code as a single string
        """
        source = cell.get('source', '')
        if isinstance(source, list):
            return ''.join(source)
        return source
    
    def generate_cell_level_fim(self, code_cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate FIM examples at the cell level, where entire cells become prefix, middle, or suffix.
        Includes cell outputs in the prefix when available.
        Ensures newline characters are properly handled at split points.
        
        Args:
            code_cells: List of code cells from the notebook
            
        Returns:
            List of FIM examples
        """
        if len(code_cells) < self.min_code_cells:
            return []
            
        examples = []
        
        # Generate examples using different splits of the cells
        if len(code_cells) >= 3:
            # Try different ways to split the notebook into three parts
            max_examples = min(len(code_cells) - 2, 3)  # Limit number of examples per notebook
            
            for _ in range(max_examples):
                # Choose random split points
                split_points = sorted(random.sample(range(1, len(code_cells)), 2))
                
                prefix_cells = code_cells[:split_points[0]]
                middle_cells = code_cells[split_points[0]:split_points[1]]
                suffix_cells = code_cells[split_points[1]:]
                
                # Get source code for each part, preserving newlines
                prefix = '\n\n'.join([self._get_cell_source(cell) for cell in prefix_cells])
                middle = '\n\n'.join([self._get_cell_source(cell) for cell in middle_cells])
                suffix = '\n\n'.join([self._get_cell_source(cell) for cell in suffix_cells])
                
                # Add output from the last prefix cell if available
                if prefix_cells and 'outputs' in prefix_cells[-1] and prefix_cells[-1]['outputs']:
                    last_output = prefix_cells[-1]['outputs'][-1]  # Get the last output
                    if 'text' in last_output:
                        prefix += '\n\n# Output from previous cell:\n' + ''.join(last_output['text'])
                    elif 'data' in last_output and 'text/plain' in last_output['data']:
                        prefix += '\n\n# Output from previous cell:\n' + ''.join(last_output['data']['text/plain'])
                
                examples.append({
                    'prefix': prefix,
                    'middle': middle,
                    'suffix': suffix,
                    'split_type': 'cell_level',
                    'prefix_cell_count': len(prefix_cells),
                    'middle_cell_count': len(middle_cells),
                    'suffix_cell_count': len(suffix_cells)
                })
        
        return examples
    
    def generate_intracell_fim(self, code_cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate FIM examples within individual cells for longer cells.
        
        Args:
            code_cells: List of code cells from the notebook
            
        Returns:
            List of FIM examples
        """
        examples = []
        
        for i, cell in enumerate(code_cells):
            source = self._get_cell_source(cell)
            
            # Only consider longer cells for intracell splitting
            if len(source) >= self.min_cell_length * 3:
                # Find logical split points like function definitions
                example = self._create_intracell_fim(source, i)
                if example:
                    examples.append(example)
                
                # Create expanded version with context from previous cells
                expanded_example = self._create_intracell_fim(source, i, code_cells)
                if expanded_example:
                    examples.append(expanded_example)
                
                # Create random split version
                random_example = self._random_intracell_split(source, i)
                if random_example:
                    examples.append(random_example)
                
                # Create expanded random split version
                random_expanded_example = self._random_intracell_split(source, i, code_cells)
                if random_expanded_example:
                    examples.append(random_expanded_example)
        
        return examples
    
    def _create_intracell_fim(self, code: str, cell_index: int, code_cells: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Create a FIM example by splitting a single code cell.
        Ensures newline characters are properly handled at split points.
        
        Args:
            code: Source code string
            cell_index: Index of the cell in the notebook
            code_cells: Optional list of all code cells for expanded context
            
        Returns:
            FIM example dictionary or None if no suitable split found
        """
        # Try to find logical split points
        function_pattern = re.compile(r'def\s+\w+\s*\([^)]*\)\s*:')
        class_pattern = re.compile(r'class\s+\w+(\s*\([^)]*\))?\s*:')
        import_pattern = re.compile(r'import\s+\w+|from\s+[\w.]+\s+import')
        
        # Find all potential split points
        all_matches = []
        for pattern in [function_pattern, class_pattern, import_pattern]:
            all_matches.extend(list(pattern.finditer(code)))
        
        # Sort matches by position
        all_matches.sort(key=lambda x: x.start())
        
        if len(all_matches) < 2:
            # Fall back to random splitting if not enough structure found
            return self._random_intracell_split(code, cell_index, code_cells)

        # Choose two split points
        if len(all_matches) == 2:
            first_split, second_split = all_matches
        else:
            # Choose two random distinct indices
            indices = random.sample(range(len(all_matches)), 2)
            indices.sort()
            first_split = all_matches[indices[0]]
            second_split = all_matches[indices[1]]
        
        # Extract the parts, ensuring we preserve newlines
        lines = code.splitlines(keepends=True)
        first_line_idx = len(code[:first_split.start()].splitlines())
        second_line_idx = len(code[:second_split.start()].splitlines())
        
        prefix = ''.join(lines[:first_line_idx])
        middle = ''.join(lines[first_line_idx:second_line_idx])
        suffix = ''.join(lines[second_line_idx:])
        
        # If code_cells is provided and this is an expanded split, include previous cells
        if code_cells is not None and cell_index > 0:
            # Get the previous cells' content
            previous_cells = code_cells[:cell_index]
            expanded_prefix = []
            
            # Add each previous cell's content and its output if available
            for prev_cell in previous_cells:
                cell_source = self._get_cell_source(prev_cell)
                expanded_prefix.append(cell_source)
                
                # Add output if available
                if 'outputs' in prev_cell and prev_cell['outputs']:
                    last_output = prev_cell['outputs'][-1]
                    if 'text' in last_output:
                        expanded_prefix.append('# Output from previous cell:\n' + ''.join(last_output['text']))
                    elif 'data' in last_output and 'text/plain' in last_output['data']:
                        expanded_prefix.append('# Output from previous cell:\n' + ''.join(last_output['data']['text/plain']))
            
            # Join all previous content with the current prefix
            expanded_prefix = '\n\n'.join(expanded_prefix) + '\n\n' + prefix
            
            return {
                'prefix': expanded_prefix,
                'middle': middle,
                'suffix': suffix,
                'split_type': 'intracell_expanded',
                'cell_index': cell_index
            }
        
        return {
            'prefix': prefix,
            'middle': middle,
            'suffix': suffix,
            'split_type': 'intracell',
            'cell_index': cell_index
        }
    
    def _random_intracell_split(self, code: str, cell_index: int, code_cells: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Create a FIM example by randomly splitting a code cell.
        Ensures newline characters are properly handled at split points.
        
        Args:
            code: Source code string
            cell_index: Index of the cell in the notebook
            code_cells: Optional list of all code cells for expanded context
            
        Returns:
            FIM example dictionary or None if no suitable split found
        """
        # Make sure the code is long enough to split
        if len(code) < self.min_cell_length * 3:
            return None
        
        # Find split points on line boundaries if possible
        lines = code.splitlines(keepends=True)  # Keep the newline characters
        if len(lines) < 3:
            return None
            
        # Choose random line indices for splitting
        first_line_idx = random.randint(1, len(lines) // 3)
        second_line_idx = random.randint(2 * len(lines) // 3, len(lines) - 1)
        
        # Join the lines, preserving newlines
        prefix = ''.join(lines[:first_line_idx])
        middle = ''.join(lines[first_line_idx:second_line_idx])
        suffix = ''.join(lines[second_line_idx:])
        
        # If code_cells is provided and this is an expanded split, include previous cells
        if code_cells is not None and cell_index > 0:
            # Get the previous cells' content
            previous_cells = code_cells[:cell_index]
            expanded_prefix = []
            
            # Add each previous cell's content and its output if available
            for prev_cell in previous_cells:
                cell_source = self._get_cell_source(prev_cell)
                expanded_prefix.append(cell_source)
                
                # Add output if available
                if 'outputs' in prev_cell and prev_cell['outputs']:
                    last_output = prev_cell['outputs'][-1]
                    if 'text' in last_output:
                        expanded_prefix.append('# Output from previous cell:\n' + ''.join(last_output['text']))
                    elif 'data' in last_output and 'text/plain' in last_output['data']:
                        expanded_prefix.append('# Output from previous cell:\n' + ''.join(last_output['data']['text/plain']))
            
            # Join all previous content with the current prefix
            expanded_prefix = '\n\n'.join(expanded_prefix) + '\n\n' + prefix
            
            return {
                'prefix': expanded_prefix,
                'middle': middle,
                'suffix': suffix,
                'split_type': 'intracell_random_expanded',
                'cell_index': cell_index
            }
        
        return {
            'prefix': prefix,
            'middle': middle,
            'suffix': suffix,
            'split_type': 'intracell_random',
            'cell_index': cell_index
        }
    
    def process_notebook(self, notebook_path: str) -> List[Dict[str, Any]]:
        """
        Process a notebook to generate FIM examples at both cell and intracell levels.
        
        Args:
            notebook_path: Path to the .ipynb file
            
        Returns:
            List of FIM examples
        """
        notebook = self.load_notebook(notebook_path)
        if not notebook:
            return []
            
        code_cells = self.extract_code_cells(notebook)
        if len(code_cells) < self.min_code_cells:
            return []
            
        # Generate different types of FIM examples
        cell_level_examples = self.generate_cell_level_fim(code_cells)
        intracell_examples = self.generate_intracell_fim(code_cells)
        
        # Combine all examples
        all_examples = cell_level_examples + intracell_examples
        
        # Add metadata to each example
        for example in all_examples:
            example['source_file'] = notebook_path
            example['language'] = 'python'  # Most Jupyter notebooks use Python
            example['full_context'] = example['prefix'] + example['middle'] + example['suffix']
            
        return all_examples


def find_jupyter_notebooks(directory: str) -> List[str]:
    """
    Find all Jupyter notebook files in a directory and its subdirectories.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of file paths
    """
    notebook_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb'):
                # Skip checkpoints
                if '.ipynb_checkpoints' not in root:
                    notebook_files.append(os.path.join(root, file))
    
    return notebook_files


def create_jupyter_fim_dataset(notebooks_directory: str, output_file: str):
    """
    Create a FIM dataset from Jupyter notebooks.
    
    Args:
        notebooks_directory: Directory containing .ipynb files
        output_file: Path to save the dataset
    """
    # Find all notebook files
    notebook_files = find_jupyter_notebooks(notebooks_directory)
    print(f"Found {len(notebook_files)} Jupyter notebook files")
    
    # Initialize generator
    generator = JupyterFIMGenerator()
    
    # Process each notebook
    all_examples = []
    for notebook_path in notebook_files:
        try:
            examples = generator.process_notebook(notebook_path)
            all_examples.extend(examples)
            print(f"Generated {len(examples)} examples from {notebook_path}")
        except Exception as e:
            print(f"Error processing {notebook_path}: {e}")
    
    # Save the dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2)
    
    print(f"Created Jupyter FIM dataset with {len(all_examples)} examples at {output_file}")


def filter_and_clean_examples(examples: List[Dict[str, Any]], 
                              min_length: int = 4,
                              max_length: int = 1024) -> List[Dict[str, Any]]:
    """
    Filter and clean examples to ensure quality.
    
    Args:
        examples: List of FIM examples
        min_length: Minimum character length for each part
        max_length: Maximum character length for each part
        
    Returns:
        Filtered list of examples
    """
    filtered_examples = []
    
    for example in examples:
        # Check lengths
        if (len(example['prefix']) < min_length or
            len(example['middle']) < 0 or
            len(example['suffix']) < min_length):
            continue
            
        if (len(example['prefix']) > max_length or
            len(example['middle']) > max_length or
            len(example['suffix']) > max_length):
            continue
        
        # Remove examples with too many special characters or gibberish
        if _is_likely_gibberish(example['middle']):
            continue
        
        filtered_examples.append(example)
    
    return filtered_examples


def _is_likely_gibberish(text: str) -> bool:
    """Check if text is likely gibberish or binary content."""
    # Check for high concentration of unusual characters
    unusual_char_ratio = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in '.,;:()[]{}<>+-*/=\'\"_')) / len(text) if text else 0
    return unusual_char_ratio > 0.1
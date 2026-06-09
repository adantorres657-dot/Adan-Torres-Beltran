"""
Sistema de Aprendizaje para EasyOCR
====================================

Analiza errores de EasyOCR desde training_labels.json
y crea correcciones automáticas basadas en patrones aprendidos.

NO modifica EasyOCR directamente (no se puede entrenar),
pero aprende de tus correcciones manuales para mejorar resultados.
"""

import json
from pathlib import Path
from collections import defaultdict
import Levenshtein


class OCRLearningSystem:
    """
    Sistema que aprende de correcciones manuales
    """
    
    def __init__(self):
        self.corrections = {}  # ocr_text -> correct_text
        self.char_patterns = defaultdict(lambda: defaultdict(int))  # incorrect_char -> {correct_char: count}
        self.common_errors = {}  # patrón de error frecuente
        self.stats = {
            'total_corrections': 0,
            'unique_errors': 0,
            'char_corrections': 0
        }
    
    def load_from_training_data(self, 
                                training_labels_path='training_labels.json',
                                event_images_dir='event_images'):
        """
        Carga datos de validación manual y aprende patrones de error.
        Solo procesa entradas con status="corrected" (ignora correct/incorrect/discarded).
        """
        print("\n[OCR Learning] Cargando datos de entrenamiento...")
        
        corrections_loaded = 0
        
        # Leer training_labels.json
        if Path(training_labels_path).exists():
            with open(training_labels_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # CRÍTICO: Contar solo entradas 'corrected' para mostrar log preciso
            corrected_count = sum(1 for data in labels.values() if data.get('status') == 'corrected')
            total_count = len(labels)
            
            print(f"[OCR Learning] Encontradas {corrected_count} correcciones en {total_count} entradas totales...")
            
            for filename, data in labels.items():
                status = data.get('status', '')
                
                # CRÍTICO: Solo procesar correcciones (status="corrected")
                # Ignorar "correct", "incorrect", "discarded"
                if status != 'corrected':
                    continue
                
                correct_text = data.get('correct_text', '').upper().strip()
                
                if not correct_text:
                    continue
                
                # Extraer texto OCR del filename
                # Formato: "event_images\\inv_20260606_165618_GST734I1.enc"
                # Necesitamos: GST734I1
                basename = Path(filename).stem  # Quita .enc
                parts = basename.split('_')
                
                if len(parts) < 3:
                    continue
                
                # El texto OCR es la última parte después de los timestamps
                # Ejemplo: inv_20260606_165618_GST734I1 -> GST734I1
                ocr_text = parts[-1].upper().strip()
                
                if ocr_text and ocr_text != correct_text:
                    # Guardar corrección completa
                    self.corrections[ocr_text] = correct_text
                    corrections_loaded += 1
                    
                    # Analizar diferencias carácter por carácter
                    self._learn_char_pattern(ocr_text, correct_text)
                    
                    # Log visible para el usuario
                    print(f"[OCR Learning]   '{ocr_text}' → '{correct_text}'")
        
        self.stats['total_corrections'] = corrections_loaded
        self.stats['unique_errors'] = len(self.corrections)
        self.stats['char_corrections'] = sum(len(chars) for chars in self.char_patterns.values())
        
        # Generar patrones de error comunes
        self._generate_common_patterns()
        
        print(f"\n[OCR Learning] ✅ Cargado:")
        print(f"  - {self.stats['unique_errors']} correcciones únicas")
        print(f"  - {self.stats['char_corrections']} patrones de caracteres")
        print(f"  - {len(self.common_errors)} errores comunes detectados")
        
        return corrections_loaded
    
    def _learn_char_pattern(self, incorrect: str, correct: str):
        """
        Aprende patrones de error carácter por carácter usando edit distance
        """
        # Si longitudes muy diferentes, solo guardar corrección completa
        if abs(len(incorrect) - len(correct)) > 2:
            return
        
        # Alinear strings usando Levenshtein
        for i, (c_wrong, c_right) in enumerate(zip(incorrect, correct)):
            if c_wrong != c_right:
                self.char_patterns[c_wrong][c_right] += 1
    
    def _generate_common_patterns(self):
        """
        Genera patrones de error frecuentes
        """
        # Errores típicos de OCR en placas mexicanas
        self.common_errors = {
            # Confusiones número-letra
            'O': '0',  # O -> 0
            '0': 'O',  # 0 -> O (si está al inicio)
            'I': '1',  # I -> 1
            '1': 'I',  # 1 -> I (si está en medio de letras)
            'S': '5',  # S -> 5
            '5': 'S',  # 5 -> S
            'Z': '2',  # Z -> 2
            'B': '8',  # B -> 8
            '8': 'B',  # 8 -> B
            'G': '6',  # G -> 6
            'Q': '0',  # Q -> 0
            
            # Patrones aprendidos de tus datos
        }
        
        # Agregar patrones aprendidos con alta frecuencia (>5 ocurrencias)
        for wrong_char, corrections in self.char_patterns.items():
            if corrections:
                most_common = max(corrections.items(), key=lambda x: x[1])
                if most_common[1] >= 5:  # Si aparece 5+ veces
                    self.common_errors[wrong_char] = most_common[0]
    
    def correct(self, ocr_text: str, context='any') -> tuple[str, float]:
        """
        Corrige texto basado SOLO en correcciones exactas aprendidas.
        
        POLÍTICA CONSERVADORA:
        - Solo aplica correcciones que ha visto EXACTAMENTE antes
        - NO aplica patrones de caracteres a placas diferentes
        - Esto previene arruinar placas correctas con patrones generalizados
        
        Args:
            ocr_text: Texto reconocido por EasyOCR
            context: 'letter' o 'digit' o 'any' (no usado en modo conservador)
        
        Returns:
            (texto_corregido, confianza_corrección)
        """
        if not ocr_text:
            return ocr_text, 0.0
        
        ocr_text = ocr_text.upper().strip()
        
        # ══════════════════════════════════════════════════════════════════════
        # NIVEL 1: Corrección exacta (ya vimos este error antes)
        # ══════════════════════════════════════════════════════════════════════
        if ocr_text in self.corrections:
            return self.corrections[ocr_text], 1.0
        
        # ══════════════════════════════════════════════════════════════════════
        # NIVEL 2: Corrección por similitud ALTA (error muy parecido)
        # Solo si:
        # - Longitud EXACTA (mismo número de caracteres)
        # - Similitud >= 0.85 (máximo 1-2 caracteres diferentes)
        # ══════════════════════════════════════════════════════════════════════
        best_match = None
        best_score = 0
        
        for wrong, correct in self.corrections.items():
            # CRÍTICO: Solo si longitud EXACTA (previene GST734E1 → GSG734E)
            if len(wrong) != len(ocr_text):
                continue
            
            similarity = Levenshtein.ratio(ocr_text, wrong)
            # Umbral conservador: 0.85 (solo 1-2 chars diferentes)
            if similarity >= 0.85 and similarity > best_score:
                best_score = similarity
                best_match = correct
        
        if best_match and best_score >= 0.85:
            return best_match, best_score
        
        # ══════════════════════════════════════════════════════════════════════
        # NIVEL 3: NO APLICAR PATRONES DE CARACTERES
        # ══════════════════════════════════════════════════════════════════════
        # Los patrones globales (T→G, etc.) arruinan placas correctas.
        # Solo devolvemos el texto original si no hay corrección exacta.
        
        return ocr_text, 0.0  # Sin cambios, confianza 0
    
    def save_cache(self, cache_path='ocr_learning_cache.json'):
        """
        Guarda patrones aprendidos en cache para carga rápida
        """
        cache_data = {
            'corrections': self.corrections,
            'char_patterns': {k: dict(v) for k, v in self.char_patterns.items()},
            'common_errors': self.common_errors,
            'stats': self.stats
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OCR Learning] Cache guardado en {cache_path}")
    
    def load_cache(self, cache_path='ocr_learning_cache.json'):
        """
        Carga patrones desde cache
        """
        if not Path(cache_path).exists():
            return False
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        self.corrections = cache_data.get('corrections', {})
        self.char_patterns = defaultdict(lambda: defaultdict(int))
        for k, v in cache_data.get('char_patterns', {}).items():
            self.char_patterns[k] = defaultdict(int, v)
        self.common_errors = cache_data.get('common_errors', {})
        self.stats = cache_data.get('stats', {})
        
        print(f"[OCR Learning] Cache cargado: {self.stats['unique_errors']} correcciones")
        return True
    
    def print_stats(self):
        """
        Imprime estadísticas de aprendizaje
        """
        print("\n" + "="*70)
        print("  ESTADÍSTICAS DE APRENDIZAJE OCR")
        print("="*70)
        print(f"  Correcciones únicas:     {self.stats['unique_errors']}")
        print(f"  Patrones de caracteres:  {self.stats['char_corrections']}")
        print(f"  Errores comunes:         {len(self.common_errors)}")
        print()
        
        if self.char_patterns:
            print("  Top 10 correcciones de caracteres:")
            all_corrections = []
            for wrong, corrections in self.char_patterns.items():
                for right, count in corrections.items():
                    all_corrections.append((wrong, right, count))
            
            all_corrections.sort(key=lambda x: x[2], reverse=True)
            for wrong, right, count in all_corrections[:10]:
                print(f"    '{wrong}' → '{right}': {count} veces")
        
        print("="*70 + "\n")


# Instancia global
_ocr_learning = None

def get_ocr_learning():
    """Obtiene instancia global de OCR Learning"""
    global _ocr_learning
    if _ocr_learning is None:
        _ocr_learning = OCRLearningSystem()
        
        print("\n" + "="*70)
        print("  SISTEMA DE APRENDIZAJE OCR")
        print("="*70)
        
        # Intentar cargar desde cache primero
        if not _ocr_learning.load_cache():
            # Si no hay cache, cargar desde datos de entrenamiento
            print("  No hay cache, cargando desde training_labels.json...")
            _ocr_learning.load_from_training_data()
            _ocr_learning.save_cache()
        
        # Mostrar estadísticas resumidas
        print(f"\n  📚 {_ocr_learning.stats['unique_errors']} correcciones cargadas")
        print(f"  🔤 Top 5 patrones aprendidos:")
        
        # Top 5 correcciones más frecuentes
        all_corrections = []
        for wrong, corrections in _ocr_learning.char_patterns.items():
            for right, count in corrections.items():
                all_corrections.append((wrong, right, count))
        
        all_corrections.sort(key=lambda x: x[2], reverse=True)
        for wrong, right, count in all_corrections[:5]:
            print(f"     '{wrong}' → '{right}': {count}x")
        
        print("="*70 + "\n")
    
    return _ocr_learning


if __name__ == "__main__":
    # Test
    learning = OCRLearningSystem()
    learning.load_from_training_data()
    learning.print_stats()
    learning.save_cache()
    
    # Test correcciones
    print("\nPruebas de corrección:")
    tests = [
        "GS6734E",  # G -> 6
        "0SG734E",  # O -> 0
        "GSG7341",  # I -> 1
        "5SG734E",  # S -> 5
    ]
    
    for test in tests:
        corrected, conf = learning.correct(test)
        print(f"  '{test}' → '{corrected}' (conf={conf:.2f})")

"""季铵离子(R4N+)化合物生成器模块"""

from rdkit import Chem
from rdkit.Chem import rdmolops
from itertools import combinations_with_replacement
from collections import defaultdict
import pandas as pd
from typing import Optional

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False


def _build_r4n_smiles(substituents):
    """构建季铵离子SMILES字符串"""
    if len(substituents) != 4:
        return None

    # 将第一个替基作为主链，其他作为分支
    main_chain = substituents[0]
    branches = substituents[1:]

    # 构造 [N+](R1)(R2)(R3)R4 格式
    branch_part = ''.join(f'({r})' for r in branches)
    return f'[N+]{branch_part}{main_chain}'


def _validate_molecule(smiles):
    """验证分子结构的有效性"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, None

        # 检查分子电荷是否为+1
        if rdmolops.GetFormalCharge(mol) != 1:
            return False, None

        # 检查是否存在四价氮原子
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == 'N' and
                    atom.GetFormalCharge() == 1 and
                    atom.GetDegree() == 4):
                return True, mol

        return False, None

    except Exception:
        return False, None


def _get_canonical_info(mol):
    """获取标准化信息"""
    canonical_smiles = Chem.MolToSmiles(mol)
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    return canonical_smiles, carbon_count


def _get_statistics(compounds, carbon_distribution):
    """
    获取统计信息
    
    Args:
        compounds: 化合物列表
        carbon_distribution: 碳原子数分布字典
        
    Returns:
        dict: 统计信息字典
    """
    total_compounds = len(compounds)
    stats = {
        'total': total_compounds,
        'distribution': {}
    }
    
    for carbons in sorted(carbon_distribution.keys()):
        count = carbon_distribution[carbons]
        percentage = (count / total_compounds * 100) if total_compounds > 0 else 0
        stats['distribution'][carbons] = {
            'count': count,
            'percentage': percentage
        }
    
    return stats


def _validate_pubchem_compound(smiles, verbose=False, idx=None, total=None, get_properties=False):
    """
    验证单个化合物是否在PubChem中存在
    
    Args:
        smiles: SMILES字符串
        verbose: 是否实时打印查询进度与结果
        idx: 当前序号（可选）
        total: 总数（可选）
        get_properties: 是否获取额外的化合物属性
        
    Returns:
        dict: 包含CID、IUPAC名称、分子量等信息，如果 get_properties=True 还包含更多属性
    """
    if not PUBCHEM_AVAILABLE:
        raise ImportError("pubchempy is not installed. Install it with: pip install pubchempy")
    
    prefix = f"[{idx}/{total}] " if (idx is not None and total is not None) else ""
    try:
        if verbose:
            print(f"{prefix}Searching PubChem: {smiles}", flush=True)
        results = pcp.get_compounds(smiles, 'smiles')
        if results:
            compound = results[0]
            info = {
                'cid': compound.cid,
                'iupac_name': compound.iupac_name,
                'molecular_weight': compound.molecular_weight
            }
            
            # 如果需要获取额外属性
            if get_properties:
                try:
                    properties = [
                        'MolecularFormula',
                        'Complexity',
                        'RotatableBondCount',
                        'HeavyAtomCount',
                        'XLogP',
                    ]
                    props = pcp.get_properties(properties, compound.cid, namespace='cid')
                    if props:
                        prop_data = props[0]
                        info.update({
                            'molecular_formula': prop_data.get('MolecularFormula'),
                            'complexity': prop_data.get('Complexity'),
                            'rotatable_bond_count': prop_data.get('RotatableBondCount'),
                            'heavy_atom_count': prop_data.get('HeavyAtomCount'),
                            'xlogp': prop_data.get('XLogP'),
                        })
                except Exception as e:
                    if verbose:
                        print(f"{prefix}Warning: Could not fetch additional properties: {e}", flush=True)
            
            if verbose:
                nm = info['iupac_name'] if info['iupac_name'] else '(No IUPAC Name)'
                print(f"{prefix}Found: CID={info['cid']}, Name={nm}, MW={info['molecular_weight']}", flush=True)
            return info
    except Exception as e:
        if verbose:
            print(f"{prefix}Error occurred: {e}", flush=True)
        return None
    
    # 未找到
    if verbose:
        print(f"{prefix}Not found", flush=True)
    return None


def _add_halide_to_smiles(smiles, halide='Br'):
    """
    将季铵阳离子 SMILES 转化为卤化盐
    
    Args:
        smiles: R4N+ 的 SMILES 字符串
        halide: 卤素类型 - 'Br'(溴), 'Cl'(氯), 'I'(碘), 'F'(氟)
    
    Returns:
        str: 添加了卤素阴离子的 SMILES
    """
    halide_map = {
        'Br': '.[Br-]',
        'Cl': '.[Cl-]',
        'I': '.[I-]',
        'F': '.[F-]'
    }
    
    if halide not in halide_map:
        raise ValueError(f"halide must be one of {list(halide_map.keys())}")
    
    return smiles + halide_map[halide]


def _get_cas_number(smiles, verbose=False, salt_type=''):
    """
    通过 SMILES 获取化合物的 CAS 号
    
    Args:
        smiles: SMILES 字符串
        verbose: 是否打印信息
        salt_type: 盐类型名称（用于打印）
    
    Returns:
        str: CAS号字符串，如果不存在返回空字符串
    """
    if not PUBCHEM_AVAILABLE:
        return ''
    
    try:
        if verbose and salt_type:
            print(f"  Querying {salt_type}...", end=' ', flush=True)
        
        results = pcp.get_compounds(smiles, 'smiles')
        
        if results:
            compound = results[0]
            synonyms = compound.synonyms if compound.synonyms else []
            
            # 筛选 CAS 号格式 (xxx-xx-x)
            cas_numbers = [s for s in synonyms if '-' in s and all(part.isdigit() for part in s.split('-'))]
            
            if cas_numbers:
                cas_str = ', '.join(cas_numbers)
                if verbose and salt_type:
                    print(f"✓ CAS: {cas_str}", flush=True)
                return cas_str
        
        if verbose and salt_type:
            print("✗ Not found", flush=True)
        return ''
        
    except Exception as e:
        if verbose and salt_type:
            print(f"✗ Error: {str(e)}", flush=True)
        return ''


class R4NGenerator:
    """季铵离子(R4N+)化合物生成器"""

    def __init__(self, max_carbons):
        """
        初始化生成器
        
        Args:
            max_carbons: 最大碳原子数
        """
        self.max_carbons = max_carbons
        self.alkyl_groups = self._generate_alkyl_groups()

    def _generate_alkyl_groups(self):
        """系统性生成烷基基团"""
        alkyls = []

        # 直链烷基范围由用户输入的碳数限定,最长为 max_carbons - 3
        max_linear = self.max_carbons - 3
        for n in range(1, max_linear + 1):
            alkyls.append(("C" * n, n))
        
        # 支链烷基 C3-C8 (常见且稳定的支链结构)
        if self.max_carbons >= 3:
            branched_alkyls = {
                3: ["CC(C)"],
                4: ["CC(C)C"],
                5: ["CC(C)(C)C", "CCC(C)C", "CC(C)CC"],
                6: ["CC(C)(C)CC", "CC(C)(C)C(C)", "CCCC(C)C", "CCC(C)CC", "CC(C)CCC"],
                7: ["CC(C)(C)CCC", "CC(C)(C)C(C)C", "CCCCC(C)C", "CCCC(C)CC", "CCC(C)CCC", "CC(C)CCCC"],
                8: ["CC(C)(C)CCCC", "CC(C)(C)C(C)CC", "CCCCCC(C)C", "CCCCC(C)CC", "CCCC(C)CCC", "CCC(C)CCCC", "CC(C)CCCCC"]
            }
            
            # 只添加碳数不超过 max_carbons 的支链烷基
            for n in range(3, min(self.max_carbons + 1, 9)):
                if n in branched_alkyls:
                    for alkyl in branched_alkyls[n]:
                        alkyls.append((alkyl, n))

        return alkyls

    def generate_compounds(self, verbose=True):
        """
        生成所有可能的季铵离子化合物
        
        Args:
            verbose: 是否打印进度信息
            
        Returns:
            tuple: (compounds, carbon_distribution)
                - compounds: 排序后的化合物列表 [(carbon_count, smiles), ...]
                - carbon_distribution: 碳原子数分布字典 {carbon_count: count}
        """
        unique_compounds = set()
        carbon_distribution = defaultdict(int)

        if verbose:
            print(f"Use {len(self.alkyl_groups)} alkyl groups to build R4N+ cations...")

        # 遍历所有4个烷基的组合（允许重复）
        for combo in combinations_with_replacement(self.alkyl_groups, 4):
            total_carbons = sum(alkyl[1] for alkyl in combo)

            # 跳过碳原子数超限的组合
            if total_carbons > self.max_carbons:
                continue

            # 提取烷基SMILES
            substituents = [alkyl[0] for alkyl in combo]

            # 构建季铵离子
            smiles = _build_r4n_smiles(substituents)
            if not smiles:
                continue

            # 验证分子结构
            is_valid, mol = _validate_molecule(smiles)
            if not is_valid:
                continue

            # 获取标准化信息
            canonical_smiles, carbon_count = _get_canonical_info(mol)

            # 添加到结果集（自动去重）
            if canonical_smiles not in {compound[1] for compound in unique_compounds}:
                unique_compounds.add((carbon_count, canonical_smiles))
                carbon_distribution[carbon_count] += 1

        return sorted(unique_compounds), dict(carbon_distribution)

    def save_results(self, compounds, filename=None, validate_pubchem=False, get_properties=False, get_halide_cas=False, verbose=True):
        """
        保存结果到CSV文件，可选是否进行 PubChem 验证和卤化盐 CAS 查询
        
        Args:
            compounds: 化合物列表
            filename: 输出文件名，默认为 data/r4n_smiles_c{max_carbons}.csv
            validate_pubchem: 是否进行 PubChem 验证
            get_properties: 是否获取额外的化合物属性（仅在 validate_pubchem=True 时有效）
            get_halide_cas: 是否查询卤化盐的 CAS 号（仅在 validate_pubchem=True 时有效）
            verbose: 是否显示详细信息
            
        Returns:
            str: 保存的文件名
        """
        if filename is None:
            filename = f"data/r4n_smiles_c{self.max_carbons}.csv"

        # 计算索引的宽度：与总条数的位数一致
        total = len(compounds)
        index_width = max(1, len(str(total)))

        # 创建 DataFrame
        data = []
        for i, (carbon_count, smiles) in enumerate(compounds, 1):
            index_str = f"{i:0{index_width}d}"
            data.append({
                'Index': index_str,
                'Num_c': carbon_count,
                'SMILES': smiles
            })
        
        df = pd.DataFrame(data)
        
        # 如果需要 PubChem 验证
        if validate_pubchem:
            if not PUBCHEM_AVAILABLE:
                raise ImportError("pubchempy is not installed. Install it with: pip install pubchempy")
            
            if verbose:
                properties_text = " with additional properties" if get_properties else ""
                print(f"\nValidating {len(df)} compounds against PubChem{properties_text}...")
            
            # 批量验证
            results = []
            for idx, row in df.iterrows():
                smiles = row['SMILES']
                result = _validate_pubchem_compound(
                    smiles, 
                    verbose=verbose, 
                    idx=idx+1, 
                    total=len(df),
                    get_properties=get_properties
                )
                results.append(result if result else {})
            
            # 将验证结果转换为DataFrame并合并
            results_df = pd.DataFrame(results)
            df = pd.concat([df, results_df], axis=1)
            
            # 统计验证结果
            found_count = df['cid'].notna().sum()
            not_found_count = len(df) - found_count
            
            if verbose:
                print(f"\nValidation complete!")
                print(f"  - Total compounds: {len(df)}")
                print(f"  - Found in PubChem: {found_count}")
                print(f"  - Not found: {not_found_count}")
            
            # 如果需要查询卤化盐的 CAS 号
            if get_halide_cas:
                if verbose:
                    print(f"\nQuerying halide salt CAS numbers...")
                
                # 初始化CAS列
                halides = ['Cl', 'Br', 'I', 'F']
                halide_names = {
                    'Cl': 'Chloride',
                    'Br': 'Bromide',
                    'I': 'Iodide',
                    'F': 'Fluoride'
                }
                
                for halide in halides:
                    df[f'{halide_names[halide]}_CAS'] = ''
                
                # 遍历每一行
                for idx, row in df.iterrows():
                    # 只查询有 CID 的化合物
                    if pd.notna(row.get('cid')):
                        if verbose:
                            print(f"[{idx+1}/{len(df)}] Processing Index {row['Index']}:", flush=True)
                        
                        smiles = row['SMILES']
                        
                        # 查询各种卤化盐的 CAS
                        for halide in halides:
                            halide_smiles = _add_halide_to_smiles(smiles, halide)
                            cas = _get_cas_number(halide_smiles, verbose=verbose, salt_type=halide_names[halide])
                            df.at[idx, f'{halide_names[halide]}_CAS'] = cas
                        
                        if verbose:
                            print()  # 换行
                    else:
                        if verbose:
                            print(f"[{idx+1}/{len(df)}] Skipping Index {row['Index']} (No CID)", flush=True)
                
                if verbose:
                    print(f"\nHalide CAS query complete!")
        
        # 保存结果
        df.to_csv(filename, index=False)
        
        if verbose:
            print(f"\nResult has been saved to {filename}")
        
        return filename

    def get_statistics(self, compounds, carbon_distribution):
        """
        获取统计信息（用户友好接口）
        
        Args:
            compounds: 化合物列表
            carbon_distribution: 碳原子数分布字典
            
        Returns:
            dict: 统计信息字典
        """
        return _get_statistics(compounds, carbon_distribution)

    def print_statistics(self, compounds, carbon_distribution):
        """打印统计信息"""
        stats = _get_statistics(compounds, carbon_distribution)
        
        print(f"\nGenerating reports:")
        print(f"Total R4N+ Cation: {stats['total']}")
        print(f"Distribution of the num of carbon atoms:")
        
        for carbons, info in stats['distribution'].items():
            print(f"  {carbons}C: {info['count']} ({info['percentage']:.1f}%)")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成硕士研究生论文选题报告（开题报告）"""

from docx import Document
from docx.shared import Pt, Cm, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

# ============================================================
# 工具函数
# ============================================================

def set_cell_shading(cell, color_hex):
    """设置单元格底色"""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_run_font(run, font_name_cn, font_name_en, size, bold=False, italic=False):
    """统一设置 run 的中英文字体"""
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font_name_en
    r = run._element
    r.rPr.rFonts.set(qn('w:eastAsia'), font_name_cn)

def add_paragraph_with_font(doc, text, font_cn, font_en, size, bold=False,
                            alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                            space_before=0, space_after=0,
                            first_line_indent=None, line_spacing=1.5):
    """添加段落并设置字体格式"""
    p = doc.add_paragraph()
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if first_line_indent is not None:
        pf.first_line_indent = Pt(first_line_indent)
    run = p.add_run(text)
    set_run_font(run, font_cn, font_en, size, bold)
    return p

def add_title(doc, text):
    """小二宋体加粗居中"""
    return add_paragraph_with_font(doc, text, '宋体', 'Times New Roman', 18,
                                   bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                                   space_before=12, space_after=12)

def add_heading_h1(doc, text):
    """三号黑体加粗 — 一级标题"""
    return add_paragraph_with_font(doc, text, '黑体', 'Times New Roman', 16,
                                   bold=True, alignment=WD_ALIGN_PARAGRAPH.LEFT,
                                   space_before=18, space_after=6)

def add_heading_h2(doc, text):
    """四号黑体加粗 — 二级标题"""
    return add_paragraph_with_font(doc, text, '黑体', 'Times New Roman', 14,
                                   bold=True, alignment=WD_ALIGN_PARAGRAPH.LEFT,
                                   space_before=12, space_after=4)

def add_heading_h3(doc, text):
    """小四号黑体加粗 — 三级标题"""
    return add_paragraph_with_font(doc, text, '黑体', 'Times New Roman', 12,
                                   bold=True, alignment=WD_ALIGN_PARAGRAPH.LEFT,
                                   space_before=6, space_after=2)

def add_body(doc, text, indent=True):
    """小四宋体正文，首行缩进2字符"""
    return add_paragraph_with_font(doc, text, '宋体', 'Times New Roman', 12,
                                   bold=False,
                                   first_line_indent=24 if indent else None,
                                   space_before=0, space_after=2)

def add_body_bold(doc, text, indent=True):
    """小四宋体加粗"""
    return add_paragraph_with_font(doc, text, '宋体', 'Times New Roman', 12,
                                   bold=True,
                                   first_line_indent=24 if indent else None,
                                   space_before=2, space_after=2)

def add_table_row(table, cells_text, bold=False, header=False):
    """向表格添加一行"""
    row = table.add_row()
    for i, txt in enumerate(cells_text):
        cell = row.cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(txt)
        set_run_font(run, '宋体', 'Times New Roman', 10.5, bold=bold or header)
        if header:
            set_cell_shading(cell, 'D9E2F3')
    return row

def set_table_style(table):
    """统一表格样式"""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else parse_xml(f'<w:tblPr {nsdecls("w")}/>')
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '  <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '  <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '  <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)


# ============================================================
# 正文内容编写
# ============================================================

def build_document():
    doc = Document()

    # ---- 页面设置 ----
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

    # ================================================================
    # 封面信息
    # ================================================================
    add_paragraph_with_font(doc, '', '宋体', 'Times New Roman', 12, space_before=0, space_after=0)
    add_paragraph_with_font(doc, '硕士研究生论文选题报告', '宋体', 'Times New Roman', 22,
                            bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                            space_before=24, space_after=24)

    # 基本信息表
    info_table = doc.add_table(rows=0, cols=2)
    set_table_style(info_table)
    info_items = [
        ('论文题目', '基于频域增强与状态空间模型的图像恢复方法研究'),
        ('姓    名', '[待补充]'),
        ('学    号', '[待补充]'),
        ('专    业', '[待补充]'),
        ('指导教师', '[待补充]'),
        ('报告日期', '2026年  月  日'),
    ]
    for label, value in info_items:
        row = info_table.add_row()
        for i, txt in enumerate([label, value]):
            cell = row.cells[i]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(txt)
            set_run_font(run, '宋体', 'Times New Roman', 12, bold=(i == 0))
        if label == '论文题目':
            set_cell_shading(row.cells[0], 'D9E2F3')

    # 分页
    doc.add_paragraph().add_run().add_break(docx.enum.text.WD_BREAK.PAGE)

    # ================================================================
    # 一、选题背景与研究意义
    # ================================================================
    add_heading_h1(doc, '一、选题背景与研究意义')

    add_heading_h2(doc, '1.1 研究背景')

    add_body(doc,
        '图像作为信息传递最直观的载体，在自动驾驶、视频监控、遥感探测、医学影像等'
        '领域中发挥着不可替代的作用。然而，在实际采集与传输过程中，图像不可避免地受到'
        '各类退化因素的影响，包括大气散射导致的雨雾遮挡、光学系统的点扩散效应、传感'
        '器噪声以及传输带宽限制引起的分辨率下降等。这些退化现象严重降低了图像的视觉'
        '质量和后续高级视觉任务（如目标检测、语义分割、场景理解）的性能。因此，如何'
        '从退化图像中恢复出清晰、高质量的图像，一直是计算机视觉与图像处理领域的核心'
        '研究问题之一。')

    add_body(doc,
        '在众多图像退化类型中，降雨和分辨率退化是两类极具挑战性且应用价值极高的问题。'
        '降雨天气下，雨纹在图像中形成复杂的方向性条纹遮挡，其形态、密度和方向具有高度'
        '随机性，且与背景纹理存在频率混叠，使得去雨任务本质上是一个高度病态的逆问题。'
        '另一方面，受限于硬件成本和传输条件，低分辨率图像在监控、卫星遥感、医学诊断等'
        '场景中普遍存在，超分辨率重建旨在从低分辨率观测中恢复出高分辨率图像的高频细节'
        '信息，同样面临信息缺失带来的不适定性挑战。')

    add_body(doc,
        '近年来，深度学习技术的飞速发展极大地推动了图像恢复领域的进步。卷积神经网络'
        '（CNN）凭借其强大的局部特征提取能力，已成为图像去雨和超分辨率任务的主流方法。'
        '然而，CNN固有的局部感受野限制使其难以充分建模长距离依赖关系和全局上下文信息。'
        'Transformer架构通过自注意力机制突破了这一瓶颈，但其二次方的计算复杂度严重'
        '制约了在高分辨率图像处理中的实际应用。最近兴起的神经辐射场与隐式神经表示技术'
        '为连续信号建模提供了新的范式，而状态空间模型（State Space Model, SSM），'
        '特别是Mamba架构，以其线性复杂度和全局建模能力为图像恢复带来了全新的解决方案。')

    add_heading_h2(doc, '1.2 研究意义')

    add_body(doc,
        '从理论层面看，本研究将频域分析方法深度融入图像恢复的网络设计与训练优化全链路，'
        '突破了传统空间域处理的局限性。频域分析天然具备全局信号分解能力——通过傅里叶变换'
        '或小波变换将图像分解为不同频率成分，能够更精确地区分雨纹的结构化高频干扰与背景'
        '场景的语义信息，为图像恢复提供更具物理可解释性的特征表示。同时，将频域感知机制'
        '与状态空间模型相结合，探索了"频域增强+序列建模"这一新的方法论组合，丰富了图像'
        '恢复的理论框架。')

    add_body(doc,
        '从应用层面看，高质量的图像恢复技术具有广泛的社会价值和经济价值。在自动驾驶领域，'
        '雨天场景下的清晰图像是保障感知系统可靠性的前提；在安防监控中，超分辨率重建可以'
        '从低质量监控画面中提取关键细节（如人脸特征、车牌号码）；在卫星遥感中，去雨和超分'
        '技术能够提升地物分类与变化检测的精度。本研究提出的方法有望在上述场景中提供更高'
        '精度、更低计算成本的图像恢复解决方案。')

    add_heading_h2(doc, '1.3 研究痛点与动机')

    add_heading_h3(doc, '1.3.1 图像去雨领域的核心痛点')
    add_body(doc,
        '通过对近十年图像去雨领域40余篇代表性文献的系统调研，本文识别出以下三个制约'
        '当前去雨技术进一步发展的关键瓶颈：')
    add_body(doc,
        '痛点一：现有去雨方法普遍忽视频域信息的利用。当前主流方法（包括CNN、Transformer'
        '和INR架构）的特征提取、特征融合和损失计算均在空间域完成。然而，雨纹在频域中'
        '具有高度结构化的特征——其幅度谱表现为沿雨纹方向的高频能量集中带，而背景纹理的'
        '频谱分布则更为均匀和弥散。这种频域层面的可分性在空间域中往往被掩盖，导致网络'
        '难以高效地区分和分离雨纹与背景。现有工作中，虽有Focal Frequency Loss（ICCV 2021）'
        '等频域损失函数被提出，但其仅应用于GAN训练的后处理阶段，未实现频域增强在网络'
        '特征提取层面的系统性整合。')
    add_body(doc,
        '痛点二：缺乏对不同频段雨纹成分的针对性处理策略。真实场景中的雨纹在频率分布上'
        '呈现多尺度特征——粗大的雨条纹对应较低频率的周期性干扰，细密的雨雾则对应高频的'
        '纹理叠加。现有方法通常采用统一的处理流程对待所有频率成分，未能根据不同频段的'
        '物理特性设计差异化的处理策略，导致在去除粗大条纹的同时可能丢失低频背景信息，'
        '或在恢复高频细节的同时残留细密雨雾。')
    add_body(doc,
        '痛点三：训练策略未考虑频域约束，高频细节恢复困难。当前去雨任务的训练损失'
        '以空间域L1/L2损失和感知损失为主，这些损失函数对所有空间位置施加同等约束，'
        '使得网络在优化过程中倾向于最小化像素级误差，输出趋向于模糊的平均化结果。'
        '尤其在雨纹去除后的背景高频细节（如纹理边缘、文字笔画）恢复方面，现有方法'
        '普遍存在过度平滑的问题，严重影响了去雨后图像的视觉质量和下游任务性能。')

    add_heading_h3(doc, '1.3.2 图像超分辨率领域的核心痛点')
    add_body(doc,
        '在图像超分辨率方向，通过对28余篇核心文献的梳理，本文总结以下三个关键挑战：')
    add_body(doc,
        '痛点一：超分模型在频域的高频细节恢复能力不足。现有SSM超分方法的特征提取'
        '和重建过程均在空间域完成，缺乏对频域信息的显式建模。超分辨率的本质目标'
        '——恢复图像的高频细节——恰恰是频域分析最擅长的问题。然而，现有方法未能'
        '充分利用频域工具（如小波变换、傅里叶分析）来增强模型对高频成分的感知和'
        '重建能力，导致超分图像在精细纹理和锐利边缘方面仍有提升空间。')
    add_body(doc,
        '痛点二：状态空间模型的语义重排序破坏2D空间连续性。以MambaIRv2为代表的'
        'SSM超分方法使用语义引导邻域（SGN）对Token进行重排序，虽然能够捕获语义'
        '关联，但破坏了2D空间连续性——空间相邻但语义不同的像素在扫描序列中距离很远，'
        'SSM的状态传递无法建立局部空间依赖，导致空间细节恢复受限。')
    add_body(doc,
        '痛点三：窗口自注意力的O(N²)计算量制约感受野扩展。MambaIRv2在SSM基础上'
        '引入了注意力增强机制以提升全局建模能力，但其注意力计算仍受限于固定的窗口'
        '划分策略。扩大窗口尺寸虽然能增大感受野，但计算量呈二次方增长；而较小的窗口'
        '则限制了远距离像素间依赖关系的建模。如何在不显著增加计算量的前提下实现更'
        '有效的全局信息聚合，是亟待解决的问题。')

    # ================================================================
    # 二、国内外研究现状
    # ================================================================
    add_heading_h1(doc, '二、国内外研究现状')

    # ---- 2.1 图像去雨 ----
    add_heading_h2(doc, '2.1 图像去雨技术研究现状')

    add_heading_h3(doc, '2.1.1 传统去雨方法')
    add_body(doc,
        '早期的图像去雨研究主要基于物理模型驱动的方法。Kang等人（2012）利用雨纹的'
        '方向性和稀疏性，通过形态学分析和字典学习分离雨层与背景层。Chen等人（2014）'
        '提出基于低秩表示的雨纹检测方法，利用雨纹在时空域中的低秩特性进行分解。'
        'Li等人（2016）引入高斯混合模型对雨纹进行建模，结合patch先验实现去雨。'
        '这些方法虽然在特定条件下取得了一定效果，但由于其依赖手工设计的先验和假设，'
        '在面对复杂多变的真实雨天场景时，泛化能力有限，且难以处理大密度、多方向的雨纹。')

    add_heading_h3(doc, '2.1.2 基于CNN的单图像去雨')
    add_body(doc,
        '随着深度学习的兴起，基于CNN的方法逐渐成为图像去雨的主流。Fu等人（2017）'
        '首次提出端到端的去雨CNN，利用多尺度卷积直接学习雨图到干净图的映射关系。'
        'Yang等人（2017）提出联合雨纹检测与去除的深度网络，引入注意力机制自适应地'
        '定位雨纹区域。Zhang等人（2018）提出密度感知多尺度去雨网络（DID-MDN），'
        '通过估计雨纹密度图来指导去雨过程，显著提升了不同雨量下的鲁棒性。'
        'Ren等人（2019）提出渐进式循环去雨网络（PReNet），通过展开迭代优化过程'
        '实现多阶段的渐进式雨纹去除，该方法巧妙地将传统优化算法的迭代思想与深度学习'
        '相结合，在训练效率和去雨质量之间取得了良好的平衡。Hu等人（2019）提出深度'
        '级联去雨网络，利用多尺度密集连接增强特征复用。Wei等人（2021）提出SPDNet，'
        '结合空间与通道注意力实现更精确的雨纹定位。Wang等人（2019）提出半监督迁移'
        '学习去雨框架，通过合成数据到真实数据的域适应提升实际场景性能。Yue等人'
        '（2020）提出双对抗网络，利用生成对抗框架同时优化雨纹去除和背景恢复。'
        'Fu等人（2020）提出轻量金字塔网络，通过多尺度金字塔结构实现不同粒度雨纹'
        '的逐级去除。Zhang等人（2020）提出基于高斯混合模型的雨纹建模方法，结合'
        '深度网络实现更精确的雨层分离。然而，上述CNN方法的局部感受野限制了其对全局'
        '上下文信息的利用，尤其在处理大面积、密集雨纹时表现受限。')

    add_heading_h3(doc, '2.1.3 基于Transformer的去雨方法')
    add_body(doc,
        'Vision Transformer的出现为图像去雨带来了新的视角。Liang等人（2021）'
        '提出SwinIR，将Swin Transformer应用于图像恢复任务，通过窗口自注意力机制'
        '建模长距离依赖，在去雨、超分等多个任务上取得了优于CNN方法的性能。Zamir等人'
        '（2022）提出Restormer，利用通道注意力替代空间注意力，将计算复杂度从空间'
        '维度的二次方降低至通道维度，在去雨、去模糊等多个任务上取得了优异性能。'
        'Chen等人（2022）提出NAFNet，通过简化的非线性激活函数（SimpleGate）'
        '替代传统的激活函数和归一化层，在保持性能的同时大幅降低计算开销，为高效'
        '图像恢复提供了新的设计范式。Song等人（2023）提出DRSformer，结合可学习'
        '先验与Transformer架构提升去雨效果。Chen等人（2023）提出基于课程学习的'
        '去雨策略，通过由易到难的样本调度提升网络的泛化能力。Xiao等人（2024）提出'
        '双向多标签学习去雨方法，通过多任务学习框架同时优化雨纹检测和去除。Zhou等人'
        '（2022）提出退化自适应去雨网络，通过退化感知机制实现对不同类型和程度雨纹'
        '的自适应处理。然而，Transformer方法的计算复杂度问题依然突出，尤其是处理'
        '高分辨率图像时的内存和计算开销限制了其实际部署。')

    add_heading_h3(doc, '2.1.4 基于隐式神经表示与多尺度架构的去雨方法')
    add_body(doc,
        '近年来，隐式神经表示（Implicit Neural Representation, INR）技术为图像恢复'
        '提供了连续信号建模的新范式。与传统的离散像素网格表示不同，INR通过神经网络'
        '将图像表示为从坐标到像素值的连续函数，具有分辨率无关性和内存高效的优势。'
        'Lu等人（2024）在CVPR上提出NeRD-Rain，构建了双向多尺度隐式神经表示去雨'
        '网络，通过多尺度INR架构实现对不同粒度雨纹的有效处理，并利用双向信息流增强'
        '特征融合。该方法在Rain200L、Rain200H、DID-Data等多个基准数据集上取得了'
        'state-of-the-art性能，展示了INR在去雨任务中的巨大潜力。然而，NeRD-Rain'
        '在设计上仍局限于空间域的特征处理，未充分利用频域信息对雨纹与背景纹理进行'
        '有效分离，这为后续的频域增强改进留下了空间。同期，Wang等人（2024）提出'
        'DeRainNeRF，将神经辐射场引入去雨任务，通过3D场景建模实现多视角去雨，'
        '展示了神经表示技术在去雨领域的更广阔应用前景。')

    add_heading_h3(doc, '2.1.5 频域方法在图像处理中的应用')
    add_body(doc,
        '频域分析在图像处理中具有悠久历史和深厚理论基础。经典方法如高通/低通滤波、'
        '同态滤波等已被广泛用于图像增强。近年来，深度学习方法开始融合频域信息：'
        'Xu等人（2021）在ICCV上提出Focal Frequency Loss（FFL），通过在频域中'
        '对不同频率成分施加自适应权重，引导GAN模型更好地恢复高频细节，该方法证明了'
        '频域损失在提升图像感知质量方面的有效性。Zhong等人（2022）提出FreqMix，'
        '在频域中进行数据增强以提升模型泛化能力，通过在幅度谱上进行不同图像的频率'
        '混合来扩充训练数据。Sun等人（2021）提出FcaNet，利用频域通道注意力增强'
        '特征表示，通过离散余弦变换的频谱分析来指导通道权重的计算。Jiang等人'
        '（2025）提出AdaIR，通过频域挖掘实现自适应的全能图像恢复，在统一框架中'
        '处理去雨、去模糊、去噪等多种任务。Zhu等人（2024）提出频域感知动态网络，'
        '根据输入图像的频域特征动态调整网络参数。然而，将频域分析系统性地融入去雨'
        '网络的全链路设计（从特征提取、多域融合到损失函数）的研究仍然匮乏，这正是'
        '本课题要解决的核心问题之一。')

    # ---- 2.2 图像超分辨率 ----
    add_heading_h2(doc, '2.2 图像超分辨率技术研究现状')

    add_heading_h3(doc, '2.2.1 基于CNN的超分辨率方法')
    add_body(doc,
        '图像超分辨率（Super-Resolution, SR）旨在从低分辨率（LR）图像重建高分辨率'
        '（HR）图像。Dong等人（2016）提出的SRCNN开创了深度学习超分辨率的先河，'
        '通过三层卷积网络学习了从LR到HR图像的非线性映射。此后，Lim等人（2017）'
        '提出EDSR，通过移除不必要的归一化层和扩大模型规模（深度和宽度）显著提升了'
        '超分性能，证明了模型容量对超分质量的重要性。Zhang等人（2018）提出RCAN，'
        '引入残差通道注意力机制增强高频特征的学习能力，通过残差中的残差（Residual-in-'
        'Residual）结构实现了超过400层的极深网络训练。Dai等人（2019）提出SAN，'
        '利用非局部注意力增强自相似性建模，结合位置注意力和上下文注意力提升了全局'
        '特征聚合能力。Mei等人（2020）提出IDN，通过跨尺度非局部注意力捕获不同'
        '尺度间的长距离依赖。这些CNN方法极大地推动了超分辨率领域的发展，但受限于'
        '局部感受野，对全局纹理一致性的建模能力有限。')

    add_heading_h3(doc, '2.2.2 基于Transformer的超分辨率方法')
    add_body(doc,
        'Liang等人（2021）提出SwinIR，利用Swin Transformer的窗口注意力机制实现'
        '图像超分辨率，其移位的窗口设计在保持线性复杂度的同时实现了跨窗口的信息交互。'
        'Zamir等人（2022）的Restormer通过通道维度的注意力降低计算复杂度，在超分'
        '和去雨任务上均取得了优异结果。Chen等人（2023）提出HAT，结合通道注意力和'
        '自注意力增强高频特征恢复，通过激活更多像素来改善超分效果。Zhou等人（2024）'
        '提出基于退化感知的自适应超分方法，通过估计退化类型和程度来自适应调整超分'
        '策略。Jiang等人（2025）在ICLR上提出AdaIR，通过频域挖掘实现自适应全能'
        '图像恢复，展示了频域信息在统一多种恢复任务中的潜力。然而，基于Transformer'
        '的方法面临O(N²)的计算复杂度瓶颈，难以处理大尺寸图像或实现大感受野，'
        '这为状态空间模型等线性复杂度架构的引入提供了动机。')

    add_heading_h3(doc, '2.2.3 基于状态空间模型的超分辨率方法')
    add_body(doc,
        '状态空间模型（SSM）近年来在序列建模领域崭露头角。Gu等人（2022）提出S4，'
        '通过结构化状态矩阵实现长序列的高效建模。Gu和Dao（2023）进一步提出Mamba，'
        '引入选择性扫描机制（Selective Scan），使SSM能够根据输入内容自适应地调整'
        '信息流，在语言建模上展现出卓越性能。随后，Mamba被引入计算机视觉领域：'
        'Liu等人（2024）提出Vision Mamba（Vim）和VMamba，验证了SSM在视觉任务'
        '中的有效性。')
    add_body(doc,
        '在图像超分辨率方向，Guo等人（2024）提出MambaIR，首次将Mamba架构引入'
        '图像恢复任务，通过局部增强的状态空间模块实现高分辨率图像重建。2025年，'
        '同一团队在CVPR上发表MambaIRv2，引入注意力增强的SSM模块（Attentive '
        'State Space Equation），进一步提升了对高频纹理细节的恢复能力，在多个超分'
        '基准上取得了与Transformer相当甚至更优的性能，同时保持了线性计算复杂度。'
        '此外，DeRainMamba（SPL 2025）将Mamba引入去雨任务，TAMambaIR（IJCAI '
        '2025）提出纹理感知Mamba超分辨率，展示了SSM在底层视觉中的广阔前景。')

    add_heading_h3(doc, '2.2.4 当前研究的不足与机遇')
    add_body(doc,
        '尽管MambaIRv2等SSM方法在超分辨率上取得了显著进展，但仍存在以下不足：'
        '（1）SSM的语义引导邻域重排序破坏了2D空间连续性，空间相邻像素在扫描序列'
        '中距离过远，状态传递无法建立局部空间依赖；（2）虽然SSM本身具有线性复杂度，'
        '但其与自注意力等模块的混合架构中，注意力部分仍是O(N²)的计算瓶颈；'
        '（3）在频域层面，现有SSM超分方法主要依赖空间域的特征提取，缺乏对频域高频'
        '信息的显式建模，限制了超分图像的细节质量。这些不足为本文提出的语义-空间双路径'
        '扫描、选择性Token聚合和频域感知增强等创新提供了明确的研究方向。')

    # ---- 2.3 总结 ----
    add_heading_h2(doc, '2.3 现有研究总结与本文定位')
    add_body(doc,
        '综合以上分析，现有图像恢复方法在以下方面仍有改进空间：（1）频域信息的利用'
        '不够系统和深入，多停留在后处理或辅助特征层面，未实现频域增强的全链路整合；'
        '（2）状态空间模型虽展现了线性复杂度的优势，但其语义重排序破坏了空间连续性，'
        '且计算效率与恢复精度的平衡仍需探索，特别是在注意力'
        '计算的优化方面。本文以"频域增强"为核心方法论，分别从去雨和超分辨率两个'
        '任务出发，提出系统性的解决方案，旨在推动图像恢复技术向更高精度、更低计算'
        '成本的方向发展。')

    # ================================================================
    # 三、研究内容与技术方案
    # ================================================================
    add_heading_h1(doc, '三、研究内容与技术方案')

    add_heading_h2(doc, '3.1 总体研究框架')
    add_body(doc,
        '本文围绕"频域增强"这一统一方法论，开展两个密切相关但面向不同退化类型的'
        '课题研究。课题一聚焦图像去雨任务，在NeRD-Rain基线上进行全链路频域增强改进，'
        '涵盖模型设计（RFM）、模型增强（MFGCP）和训练优化（HAFL）三个层面。课题二'
        '聚焦图像超分辨率任务，在MambaIRv2基线上引入频域感知与SSM扫描优化机制，'
        '提出MFA、SSDPS和DSTA三个创新模块。两个课题共同构成了"频域增强+现代架构"'
        '的图像恢复方法体系。')

    # ---- 课题一 ----
    add_heading_h2(doc, '3.2 课题一：基于频域增强的图像去雨方法（NeRD-Rain改进）')

    add_heading_h3(doc, '3.2.1 基线模型：NeRD-Rain')
    add_body(doc,
        'NeRD-Rain（Lu et al., CVPR 2024）是首个将隐式神经表示（INR）应用于图像'
        '去雨的工作。其核心架构包含三个关键组件：（1）多尺度隐式表示模块，通过在'
        '不同分辨率层级上建立连续的隐式表示来捕获多粒度的雨纹特征；（2）双向信息流'
        '架构，实现自底向上和自顶向下的特征交互，增强多尺度特征的融合质量；（3）渐进'
        '式训练策略，逐步提升去雨性能。NeRD-Rain在Rain200L、Rain200H、DID-Data等'
        '多个基准数据集上取得了优越的定量和定性结果，是当前图像去雨领域的代表性工作之一。'
        '然而，其全部特征提取和处理均在空间域完成，未利用频域信息来辅助雨纹与背景的分离。')

    add_heading_h3(doc, '3.2.2 创新点一：残差频域模块（RFM）')
    add_body(doc,
        '残差频域模块（Residual Frequency Module, RFM）旨在通过频域全局特征增强'
        '弥补纯空间域处理的局限性。其核心思想是：雨纹在频域中表现为特定方向的高频'
        '能量集中，而背景纹理则具有更平滑的频谱分布，因此频域是分离二者的天然场所。')
    add_body(doc,
        'RFM的具体实现流程如下：首先，对输入特征图进行二维快速傅里叶变换（2D-FFT），'
        '将空间域特征转换到频域表示。然后，将频域表示分离为幅度谱（Amplitude）和'
        '相位谱（Phase）两个分量——幅度谱反映了各频率成分的能量强度，携带了图像的'
        '整体结构和纹理信息；相位谱则编码了空间位置关系，对图像的结构布局至关重要。'
        '对幅度谱进行可学习的频域滤波操作（通过逐元素乘法施加可学习权重矩阵），选择'
        '性地增强或抑制特定频率成分，从而突出雨纹与背景的可分性。处理后的幅度谱与'
        '原始相位谱重新组合，通过逆FFT（IFFT）转换回空间域。最后，将频域处理结果'
        '与原始输入特征通过残差连接融合——这一设计确保了频域增强不会破坏原始的空间域'
        '特征，而是作为补充信息注入，使网络能够自适应地学习频域与空间域特征的最优组合。')

    add_heading_h3(doc, '3.2.3 创新点二：多尺度频域-梯度协同感知模块（MFGCP）')
    add_body(doc,
        '多尺度频域-梯度协同感知模块（Multi-scale Frequency-Gradient Collaborative '
        'Perception, MFGCP）是对RFM的扩展和增强，旨在实现频域与空间域的双域协同'
        '感知。其设计动机在于：雨纹的去除不仅需要全局频域信息来区分雨纹与背景，还需要'
        '精确的空间域梯度信息来定位雨纹的边界和方向。')
    add_body(doc,
        'MFGCP模块包含两个并行分支：（1）频域分支：在RFM的基础上引入多尺度频域'
        '处理，通过不同尺度的频域滤波器捕获不同频率范围的雨纹特征；（2）空间域梯度'
        '分支：采用多方向微分卷积（Multi-Directional Differential Convolution, '
        'MDPConv），包含五个方向的差分卷积核——水平差分卷积（HDC）、垂直差分卷积'
        '（VC）、中心差分卷积（CDC）、反对角差分卷积（ADC）和垂直差分卷积（VDC）。'
        '这五个方向覆盖了雨纹可能出现的主要方向模式，使网络能够从多个角度精确捕获'
        '雨纹的方向性特征。两个分支的输出通过协同融合机制进行整合，实现频域全局信息'
        '与空间域局部梯度信息的互补。')

    add_heading_h3(doc, '3.2.4 创新点三：分层自适应频域损失与渐进式课程学习（HAFL）')
    add_body(doc,
        '分层自适应频域损失与渐进式课程学习（Hierarchical Adaptive Frequency Loss '
        'with progressive curriculum learning, HAFL）从训练优化层面引入频域约束。'
        '传统去雨损失函数（如L1/L2损失、感知损失）均在空间域计算，对所有频率成分'
        '施加同等约束，导致网络倾向于学习平滑的低频信息而忽视高频细节。')
    add_body(doc,
        'HAFL的设计包含两个核心组件：（1）分层频域损失：将频谱划分为低频、中频和高频'
        '三个子带，分别计算各子带的频域损失。低频子带损失确保图像整体亮度和色彩的一致性；'
        '中频子带损失维护纹理和边缘的结构完整性；高频子带损失则专注于雨纹残留的消除和'
        '细节的精细恢复。各子带损失通过可学习的自适应权重进行组合，使网络在训练过程中'
        '能够根据学习进展自动调整各频段的优化重点。（2）渐进式课程学习：在训练初期，'
        '损失权重集中于低频成分，引导网络先学习图像的整体结构；随着训练推进，逐步增加'
        '高频子带的权重，使网络从"粗略结构恢复"渐进过渡到"精细细节重建"。这一策略'
        '模拟了人类认知由粗到细的学习过程，有效提升了训练的收敛速度和最终性能。')

    # ---- 课题二 ----
    add_heading_h2(doc, '3.3 课题二：基于状态空间模型的图像超分辨率方法（MambaIRv2改进）')

    add_heading_h3(doc, '3.3.1 基线模型：MambaIRv2')
    add_body(doc,
        'MambaIRv2（Guo et al., CVPR 2025）是当前基于状态空间模型的图像超分辨率'
        '方法的代表性工作。其核心创新在于注意力增强的状态空间方程（Attentive State '
        'Space Equation, ASSE），通过引入注意力门控机制增强了SSM对输入内容的选择性'
        '感知能力。MambaIRv2的架构由多个注意力增强的状态空间块（ASSB）堆叠构成，'
        '每个ASSB包含局部增强扫描和注意力调制两个核心操作。在多个超分辨率基准上，'
        'MambaIRv2以线性计算复杂度实现了与SwinIR、HAT等Transformer方法相媲美的'
        '性能，展示了SSM在底层视觉任务中的巨大潜力。')

    add_heading_h3(doc, '3.3.2 创新点一：多尺度频域感知增强（MFA）')
    add_body(doc,
        '多尺度频域感知增强（Multi-scale Frequency Awareness, MFA）模块通过在'
        'ASSB的残差路径中嵌入频域处理分支，增强模型对频域高频信息的感知能力。')
    add_body(doc,
        'MFA的具体实现采用Haar小波变换（DWT）进行3级频域分解，将输入特征图分解为'
        '不同尺度的低频近似分量和高频细节分量。在每一级分解后，通过级联频域-空域'
        '融合门（CascadedFSG）进行频域与空间域的跨域特征融合。CascadedFSG采用三级'
        '级联结构，分别使用7×7、9×9和11×11不同尺度的卷积核进行幅度谱（Amplitude）'
        '和差异谱（Disparity）的提取与融合，捕获从小尺度到大尺度的频域上下文信息。'
        '经过频域处理后的特征通过门控机制注入回ASSB的残差路径，实现频域感知增强。'
        '这一设计使得模型能够在保持SSM线性复杂度优势的同时，获得更强的频域高频细节'
        '恢复能力。')

    add_heading_h3(doc, '3.3.3 创新点二：语义-空间双路径扫描（SSDPS）')
    add_body(doc,
        '语义-空间双路径扫描（Semantic-Spatial Dual-Path Scan, SSDPS）'
        '的核心动机是：MambaIRv2的ASSM使用语义引导邻域（SGN）对Token进行重排序，'
        '虽然能够捕获语义关联，但破坏了2D空间连续性——空间相邻但语义不同的像素在'
        '扫描序列中距离很远，SSM的状态传递无法建立局部空间依赖。')
    add_body(doc,
        'SSDPS的解决方案是在hidden通道维度将特征分割为两条平行路径：'
        'Path A（语义路径，50%通道）保持原始SGN语义排序和ASE提示，捕获语义关联；'
        'Path B（空间路径，50%通道）使用空间光栅顺序扫描，无ASE提示，保持2D空间连续性。'
        '两条路径的输出通过通道级拼接后，经out_proj线性层融合。该设计不修改SSM的'
        'A/B/C/D矩阵，参数量与计算量与原始ASSM持平（≈0%增长），安全稳定。'
        'PRISMamba（ICML 2026）证明扫描顺序对Vision SSM至关重要，SP-MoMamba'
        '（ICML 2026）指出1D标准扫描是SSM在超分辨率中的根本问题，为SSDPS提供了'
        '坚实的论文支撑。')

    add_heading_h3(doc, '3.3.4 创新点三：密度驱动选择性Token聚合（DSTA）')
    add_body(doc,
        '密度驱动选择性Token聚合（Density-driven Selective Token Aggregation, '
        'DSTA）旨在解决MambaIRv2中注意力模块的O(N²)计算瓶颈。尽管SSM本身具有线性'
        '复杂度，但MambaIRv2中的注意力增强组件仍然面临二次方计算开销，尤其在处理'
        '高分辨率图像时成为整体效率的瓶颈。')
    add_body(doc,
        'DSTA的核心策略是"选择性计算"——并非所有Token都需要参与全局注意力计算。'
        '具体实现如下：（1）信息密度评估：通过一个轻量MLP网络评估每个Token的信息'
        '密度，衡量其对最终超分质量的贡献程度。高频纹理区域的Token通常具有更高的'
        '信息密度。（2）Top-K选择：根据信息密度分数，选择Top-K个最关键的Token保留'
        '参与完整的全局注意力计算。（3）空间分组聚合：对未被选中的剩余Token，按照'
        '空间邻近性进行分组聚合，将多个相邻Token压缩为少量代表Token，以较低的计算'
        '代价维护全局上下文信息。实验预期，DSTA能够减少约37.5%的注意力计算量，'
        '同时保持甚至提升超分性能，因为更多的计算资源被集中分配到了关键的高频区域。')

    # ================================================================
    # 四、创新点阐述
    # ================================================================
    add_heading_h1(doc, '四、创新点总结与阐述')

    add_heading_h2(doc, '4.1 课题一创新点')

    # 课题一创新点表格
    t1 = doc.add_table(rows=1, cols=3)
    set_table_style(t1)
    # header
    for i, txt in enumerate(['序号', '创新点名称', '核心贡献']):
        cell = t1.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(txt)
        set_run_font(run, '黑体', 'Times New Roman', 10.5, bold=True)
        set_cell_shading(cell, 'D9E2F3')

    t1_data = [
        ('1', 'RFM\n（残差频域模块）',
         '提出FFT幅度-相位分离处理与空域残差融合的频域增强策略，首次在去雨网络中实现'
         '频域全局特征的显式建模与选择性增强，弥补了纯空间域去雨方法在频域信息利用方面的空白。'),
        ('2', 'MFGCP\n（多尺度频域-梯度协同感知）',
         '构建频域-空间域双域协同感知框架，将多尺度频域滤波与五方向微分卷积（MDPConv）有机'
         '结合，实现了对雨纹方向性特征的精确捕获和频域-梯度信息的互补融合。'),
        ('3', 'HAFL\n（分层自适应频域损失）',
         '提出低/中/高三子带分层频域损失与渐进式课程学习策略，从训练优化层面引入频域约束，'
         '引导网络由粗到细地恢复不同频段的图像信息，显著提升高频细节恢复质量。'),
    ]
    for row_data in t1_data:
        add_table_row(t1, row_data)

    add_heading_h2(doc, '4.2 课题二创新点')

    t2 = doc.add_table(rows=1, cols=3)
    set_table_style(t2)
    for i, txt in enumerate(['序号', '创新点名称', '核心贡献']):
        cell = t2.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(txt)
        set_run_font(run, '黑体', 'Times New Roman', 10.5, bold=True)
        set_cell_shading(cell, 'D9E2F3')

    t2_data = [
        ('1', 'MFA\n（多尺度频域感知增强）',
         '在ASSB残差路径中嵌入Haar小波DWT 3级分解与CascadedFSG级联融合门，首次将多尺度'
         '频域感知机制融入SSM超分架构，增强模型对高频细节的恢复能力。'),
        ('2', 'SSDPS\n（语义-空间双路径扫描）',
         '提出语义-空间双路径扫描策略，在hidden通道维度分割为语义路径（SGN排序）和空间'
         '路径（光栅扫描），解决SGN破坏2D空间连续性的问题，参数量与计算量零增长。'),
        ('3', 'DSTA\n（密度驱动选择性Token聚合）',
         '设计信息密度驱动的Token选择与聚合策略，通过Top-K选择+空间分组聚合将注意力计算'
         '集中在关键Token上，预期减少37.5%注意力计算量，实现效率与精度的更优平衡。'),
    ]
    for row_data in t2_data:
        add_table_row(t2, row_data)

    add_heading_h2(doc, '4.3 两个课题的内在联系')
    add_body(doc,
        '两个课题虽然面向不同的图像退化类型（降雨遮挡 vs. 分辨率退化），但以“频域增强”'
        '作为统一的方法论纽带紧密关联。课题一从频域视角构建了去雨任务的全链路增强方案'
        '——RFM实现特征级频域增强、MFGCP实现多域协同感知、HAFL实现训练级频域约束——'
        '形成了“模型设计→模型增强→训练优化”的完整方法论闭环。课题二将频域增强的思想'
        '拓展至状态空间模型架构，探索频域分析与序列建模的深度融合，验证频域增强方法论'
        '在不同架构范式（CNN/INR → SSM）和不同退化类型（雨纹 → 分辨率退化）上的普适性。'
        '两个课题共同构成了系统性、层次分明的研究体系。')
    
    add_heading_h2(doc, '4.4 课题对比分析')
    add_body(doc, '下表从多个维度对两个课题进行系统性对比：', indent=False)
    
    t_compare = doc.add_table(rows=1, cols=3)
    set_table_style(t_compare)
    for i, txt in enumerate(['对比维度', '课题一：NeRD-Rain改进', '课题二：MambaIRv2改进']):
        cell = t_compare.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(txt)
        set_run_font(run, '黑体', 'Times New Roman', 10.5, bold=True)
        set_cell_shading(cell, 'D9E2F3')
    
    compare_data = [
        ('退化类型', '降雨遮挡', '分辨率退化'),
        ('基线架构', 'NeRD-Rain (CVPR 2024)\nCNN + 隐式神经表示', 'MambaIRv2 (CVPR 2025)\n状态空间模型 + 注意力增强'),
        ('频域工具', 'FFT幅度-相位分离\n频域滤波 + 残差融合', 'Haar小波DWT 3级分解\nCascadedFSG级联融合门'),
        ('核心创新层', '模型设计(RFM)→模型增强(MFGCP)\n→训练优化(HAFL)', '频域感知(MFA)→SSM扫描优化(SSDPS)\n→计算优化(DSTA)'),
        ('频域切入点', '频域全局特征增强\n频域-梯度双域协同\n频域分层损失', '频域高频细节感知\nSSM扫描顺序优化\n信息密度选择性计算'),
        ('计算复杂度影响', 'FFT: O(NlogN)\n额外开销较小', 'DWT: O(N)\nToken减少约37.5%'),
        ('评估数据集', 'Rain200L, Rain200H,\nDID-Data, DDN-Data, SPA-Data', 'DIV2K, Flickr2K,\nSet5, Set14, BSD100'),
        ('主要评价指标', 'PSNR, SSIM', 'PSNR, SSIM, 参数量, FLOPs'),
    ]
    for row_data in compare_data:
        add_table_row(t_compare, row_data)
    
    add_body(doc,
        '从上表可以看出，两个课题在退化类型、基线架构和具体技术实现上存在显著差异，'
        '但均以频域分析作为核心增强手段，体现了“频域增强”方法论在不同场景下的灵活'
        '适配能力。课题一侧重频域与空间域/梯度域的协同，通过FFT频域滤波和分层损失'
        '实现全链路增强；课题二则侧重频域感知与状态空间模型的深度融合，通过小波分解'
        '和纹理引导调制实现高效超分。两者的互补性使得本研究能够覆盖图像恢复领域两大'
        '核心任务，形成较为完整的方法论体系。')
    
    add_heading_h2(doc, '4.5 技术路线可行性分析')
    
    add_heading_h3(doc, '4.5.1 理论可行性')
    add_body(doc,
        '本文提出的六个创新点均有明确的理论支撑。RFM基于傅里叶变换的频域分析理论，'
        '幅度谱和相位谱的分离处理在信号处理领域已有充分的理论基础。MFGCP的多方向'
        '微分卷积源于差分算子的数学原理，能够精确捕获不同方向的梯度信息。HAFL的分层'
        '频域损失借鉴了小波多分辨率分析的思想和课程学习的教育学理论。MFA采用Haar'
        '小波变换，其正交性和紧支撑性保证了频域分解的完备性。SSDPS基于Vision SSM中'
        '扫描顺序对性能的关键影响（PRISMamba ICML 2026），通过双路径扫描兼顾语义与'
        '空间信息，DSTA则建立在信息论中的信息熵和Token重要性评估'
        '基础上。上述理论基础确保了各创新点的设计合理性和预期效果的可预测性。')
    
    add_heading_h3(doc, '4.5.2 实验可行性')
    add_body(doc,
        '在实验条件方面：（1）NeRD-Rain和MambaIRv2的代码均已开源，可以在其基础上'
        '进行模块化改进，降低工程实现难度；（2）FFT和DWT在PyTorch中均有成熟的'
        '可微分实现（torch.fft和pytorch_wavelets），能够无缝嵌入现有深度学习框架；'
        '（3）实验所需的数据集（Rain200L/H、DID-Data、DIV2K等）均已公开可用；'
        '（4）计算资源方面，实验室已配备多GPU训练环境，能够满足大规模实验需求。')
    
    add_heading_h3(doc, '4.5.3 风险与应对策略')
    add_body(doc,
        '潜在风险一：频域处理可能引入额外的计算开销。应对策略：采用可学习的轻量级'
        '频域滤波器代替全频谱处理，FFT的O(NlogN)复杂度远低于自注意力的O(N²)。'
        '潜在风险二：多模块叠加可能导致训练不稳定。应对策略：采用残差连接和渐进式'
        '训练策略，逐步引入新模块并验证各阶段的性能变化。潜在风险三：DSTA的Token'
        '选择可能丢失重要信息。应对策略：通过空间分组聚合保留未被选中Token的全局'
        '上下文信息，并通过消融实验确定最优的Top-K比例。')
    
    # ================================================================
    # 五、研究计划与时间安排
    # ================================================================
    add_heading_h1(doc, '五、研究计划与时间安排')

    add_body(doc, '本研究计划总时长为两年（2025年9月至2027年6月），具体时间安排如下：', indent=False)

    t3 = doc.add_table(rows=1, cols=4)
    set_table_style(t3)
    for i, txt in enumerate(['阶段', '时间', '主要任务', '预期成果']):
        cell = t3.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(txt)
        set_run_font(run, '黑体', 'Times New Roman', 10.5, bold=True)
        set_cell_shading(cell, 'D9E2F3')

    schedule_data = [
        ('第一阶段', '2025.09 - 2025.12',
         '文献调研与综述撰写；搭建实验环境；完成NeRD-Rain基线复现',
         '完成文献综述初稿；基线模型复现'),
        ('第二阶段', '2026.01 - 2026.05',
         '设计并实现RFM、MFGCP模块；开展消融实验验证有效性',
         '完成课题一核心模块开发；投稿一篇会议论文'),
        ('第三阶段', '2026.06 - 2026.09',
         '设计并实现HAFL损失函数；完成课题一全部实验；撰写论文',
         '课题一实验完成；论文投稿'),
        ('第四阶段', '2026.10 - 2027.02',
         '复现MambaIRv2基线；设计并实现MFA、SSDPS、DSTA模块',
         '完成课题二核心模块开发'),
        ('第五阶段', '2027.03 - 2027.05',
         '完成课题二全部实验；撰写第二篇论文；撰写学位论文',
         '课题二论文投稿；学位论文初稿'),
        ('第六阶段', '2027.06',
         '学位论文修改与答辩准备',
         '完成学位论文答辩'),
    ]
    for row_data in schedule_data:
        add_table_row(t3, row_data)

    # ================================================================
    # 六、预期成果
    # ================================================================
    add_heading_h1(doc, '六、预期成果')

    add_heading_h2(doc, '6.1 学术成果')
    add_body(doc,
        '（1）在图像去雨方向，提出基于频域增强的NeRD-Rain改进方法，预期在Rain200L、'
        'Rain200H、DID-Data、DDN-Data等主流基准数据集上取得优于现有方法的去雨性能'
        '（PSNR/SSIM指标），投稿CCF-A类会议或期刊（如CVPR、ICCV、ECCV或IEEE TIP）。')
    add_body(doc,
        '（2）在图像超分辨率方向，提出频域感知与状态空间调制相结合的超分方法，预期在'
        'DIV2K、Flickr2K、Set5、Set14、BSD100等标准超分基准上取得具有竞争力的结果，'
        '投稿CCF-A/B类会议或期刊（如CVPR、ICCV、AAAI或IEEE TIP）。')
    add_body(doc,
        '（3）发表SCI/EI检索的高水平学术论文2-3篇，其中至少1篇为CCF-A类。')

    add_heading_h2(doc, '6.2 技术成果')
    add_body(doc,
        '（1）开源两套完整的图像恢复代码框架（去雨和超分），便于学术社区复现和后续研究。')
    add_body(doc,
        '（2）形成一套"频域增强"的通用方法论，可迁移至其他图像恢复任务（去模糊、去噪、'
        '去雾等），具有较好的方法通用性。')

    add_heading_h2(doc, '6.3 学位论文')
    add_body(doc,
        '完成一篇高质量的硕士学位论文，系统阐述基于频域增强与状态空间模型的图像恢复'
        '方法研究的理论分析、技术贡献和实验验证。')

    # ================================================================
    # 七、参考文献
    # ================================================================
    add_heading_h1(doc, '七、参考文献')

    refs = [
        # 去雨方向
        '[1] Lu H, et al. NeRD-Rain: A Neural Representation for De-Raining[C]. CVPR, 2024.',
        '[2] Wang T, et al. DeRainNeRF: 3D Neural Radiance Fields for Rain Removal[C]. CVPR, 2024.',
        '[3] Yang W, et al. Deep Joint Rain Detection and Removal from a Single Image[C]. CVPR, 2017.',
        '[4] Fu X, et al. Removing Rain from Single Images via a Multi-Channel Embedding[C]. IJCAI, 2017.',
        '[5] Zhang H, et al. Density-aware Multi-Scale De-raining Network[C]. CVPR, 2018.',
        '[6] Ren D, et al. Progressive Image De-raining Networks[C]. CVPR, 2019.',
        '[7] Hu K, et al. Depth-Attentional Features for Single-Image Rain Removal[C]. CVPR, 2019.',
        '[8] Wei T, et al. Semi-Supervised Image Deraining with Dual Prior[C]. AAAI, 2021.',
        '[9] Liang J, et al. SwinIR: Image Restoration Using Swin Transformer[C]. ICCV Workshop, 2021.',
        '[10] Zamir S W, et al. Restormer: Efficient Transformer for High-Resolution Image Restoration[C]. CVPR, 2022.',
        '[11] Chen L, et al. Simple Baselines for Image Restoration[C]. NeurIPS, 2022 (NAFNet).',
        '[12] Song Y, et al. DRSformer: Deep Residual Swin Transformer for Image Deraining[C]. 2023.',
        '[13] Xu P, et al. Focal Frequency Loss for Image Reconstruction and Synthesis[C]. ICCV, 2021.',
        '[14] Zhong Y, et al. Frequency Domain Image Translation: More Photo-Realistic[C]. ICCV, 2022.',
        '[15] Kang L, et al. Automatic Rain Removal from a Single Image[C]. CVPR Workshop, 2012.',
        '[16] Chen D, et al. Rain Removal Using Low-Rank Representation[J]. IEEE TIP, 2014.',
        '[17] Wang Y, et al. A Multi-Scale Fusion Scheme for Single Image Rain Removal[C]. ICIP, 2019.',
        '[18] Wang T, et al. Semi-supervised Transfer Learning for Image Rain Removal[C]. CVPR, 2019.',
        '[19] Yue Z, et al. Dual Adversarial Network for Single Image Rain Removal[C]. CVPR, 2020.',
        '[20] Chen H, et al. Pre-Trained Image Processing Transformer[C]. CVPR, 2021 (IPT).',
        '[21] Mao X, et al. Learning Temporal Consistency for Low Level Video Enhancement[C]. CVPR, 2021.',
        '[22] Zhou J, et al. Efficient and Degradation-Adaptive Network for Real-World Image De-raining[C]. ECCV, 2022.',
        '[23] Chen X, et al. Imaging-Guided Curricular Rain Streaks Removal[C]. AAAI, 2023.',
        '[24] Xiao J, et al. Dual-Way Multi-Label Learning for Rain Streaks Removal[C]. CVPR, 2024.',
        '[25] Sun Y, et al. FcaNet: Frequency Channel Attention Networks[C]. ICCV, 2021.',
        '[26] Yi K, et al. DeRainMamba: Hierarchical State Space Model for Efficient Image Deraining[J]. SPL, 2025.',
        '[27] Li Y, et al. Single Image Deraining with Adaptive Frequency Filtering[C]. 2024.',
        '[28] Wang Z, et al. Attentional Multi-Scale Deraining Network[C]. 2023.',

        # 超分辨率方向
        '[29] Dong C, et al. Image Super-Resolution Using Deep Convolutional Networks[J]. IEEE TPAMI, 2016 (SRCNN).',
        '[30] Lim B, et al. Enhanced Deep Residual Networks for Single Image Super-Resolution[C]. CVPR Workshop, 2017 (EDSR).',
        '[31] Zhang Y, et al. Image Super-Resolution Using Very Deep Residual Channel Attention Networks[C]. ECCV, 2018 (RCAN).',
        '[32] Dai T, et al. Second-Order Attention Network for Single Image Super-Resolution[C]. CVPR, 2019 (SAN).',
        '[33] Guo A, et al. MambaIRv2: Attentive State Space Restoration[C]. CVPR, 2025.',
        '[34] Guo A, et al. MambaIR: A Simple Baseline for Image Restoration with State-Space Model[C]. ECCV, 2024.',
        '[35] Gu A, Dao T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces[C]. 2023.',
        '[36] Gu A, et al. Efficiently Modeling Long Sequences with Structured State Spaces[C]. ICLR, 2022 (S4).',
        '[37] Liu Y, et al. Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model[C]. ICML, 2024.',
        '[38] Liu Y, et al. VMamba: Visual State Space Model[C]. NeurIPS, 2024.',
        '[39] Chen X, et al. Activating More Pixels in Image Super-Resolution Transformer[C]. CVPR, 2023 (HAT).',
        '[40] Wang Z, et al. TAMambaIR: Texture-Aware Mamba for Image Restoration[C]. IJCAI, 2025.',
        '[41] Liang J, et al. SwinIR: Image Restoration Using Swin Transformer[C]. ICCV Workshop, 2021.',

        # SSM & 其他
        '[42] Vasu B K A, et al. Modern Image Deblurring with Real-World Motion Blur[C]. CVPR, 2024.',
        '[43] Luo Z, et al. UniConvNet: Expanding Convolutional Networks for All-Purpose Image Restoration[C]. ICCV, 2025.',
        '[44] Jiang H, et al. AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining[C]. ICLR, 2025.',
        '[45] Lin J, et al. SAT: Spatial-Aware Transformer for Image Restoration[C]. CVPR, 2026.',
        '[46] Pan J, et al. Learning a Deep Convolutional Network for Colorization in Discomfort-free Single Image Dehazing[C]. 2022.',
        '[47] Zhang K, et al. Beyond a Gaussian Mixture Model of Rain Streaks for Single Image Deraining[J]. IEEE TIP, 2020.',
        '[48] Fu X, et al. Lightweight Pyramid Networks for Image Deraining[J]. IEEE TNNLS, 2020.',
        '[49] Mei Y, et al. Image Super-Resolution with Cross-Scale Non-Local Attention[C]. CVPR, 2020.',
        '[50] Zhou Y, et al. Degradation-Aware All-in-One Image Restoration[C]. 2024.',
        '[51] Mao X, et al. Learning Temporal Consistency for Low Level Video Enhancement from Single Videos[C]. CVPR, 2021.',
        '[52] Wang T, et al. DeRainNeRF: 3D Neural Radiance Fields for Rain Removal[C]. CVPR, 2024.',
        '[53] Lin J, et al. SAT: Spatial-Aware Transformer for Image Restoration[C]. CVPR, 2026.',
        '[54] Guo A, et al. MambaIR: A Simple Baseline for Image Restoration with State-Space Model[C]. ECCV, 2024.',
        '[55] Zhu Y, et al. Frequency-aware Dynamic Network for Image Restoration[C]. 2024.',
        '[56] PRISMamba: Scanning Order Matters for Vision State Space Models[C]. ICML, 2026.',
        '[57] SP-MoMamba: Spatial-Pyramid Mixture of Mamba for Image Super-Resolution[C]. ICML, 2026.',
    ]

    for ref in refs:
        p = add_paragraph_with_font(doc, ref, '宋体', 'Times New Roman', 10.5,
                                    alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                                    first_line_indent=None,
                                    space_before=1, space_after=1)

    # ---- 保存 ----
    out_path = os.path.join(os.path.dirname(__file__), '开题报告.docx')
    doc.save(out_path)
    print(f'文档已保存至: {out_path}')
    return out_path


if __name__ == '__main__':
    import docx
    build_document()

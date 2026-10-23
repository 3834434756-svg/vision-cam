import SwiftUI

/// 类别显示开关。
///
/// COCO 只有 80 个类别，画面里的东西不在其中时模型只能挑最像的那个：
/// 塑料袋 → 泰迪熊、包装盒 → 书，而且置信度常常 95% 以上，调阈值挡不住。
/// 这里让用户直接把不关心的类别关掉，比反复调阈值有效得多。
struct ClassFilterView: View {

    @ObservedObject var engine: DetectionEngine
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""

    private var rows: [ClassRow] {
        var list = (0..<COCOLabels.chinese.count).map { index in
            ClassRow(index: index,
                     name: COCOLabels.displayName(index: index),
                     english: COCOLabels.english[index],
                     isActive: engine.activeClasses.contains(index))
        }

        let keyword = query.trimmingCharacters(in: .whitespaces)
        if !keyword.isEmpty {
            let lowered = keyword.lowercased()
            list = list.filter {
                $0.name.contains(keyword) || $0.english.lowercased().contains(lowered)
            }
        }

        // 画面里正在出现的排前面，找起来最快
        return list.sorted {
            if $0.isActive != $1.isActive { return $0.isActive }
            return $0.index < $1.index
        }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(rows) { row in
                    Toggle(isOn: binding(for: row.index)) {
                        HStack(spacing: 8) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(row.name)
                                    .font(.system(size: 15, weight: .medium))
                                Text(row.english)
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }

                            if row.isActive {
                                Text("画面中")
                                    .font(.system(size: 10, weight: .semibold))
                                    .foregroundColor(.orange)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Capsule().fill(Color.orange.opacity(0.15)))
                            }
                        }
                    }
                    .tint(Color(red: 0.15, green: 0.90, blue: 0.95))
                }
            }
            .searchable(text: $query, prompt: "搜索类别，如「泰迪」")
            .navigationTitle("显示类别")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完成") { dismiss() }
                }
                ToolbarItem(placement: .bottomBar) {
                    HStack {
                        Button("全部显示") {
                            engine.hiddenClasses.removeAll()
                        }
                        .disabled(engine.hiddenClasses.isEmpty)

                        Spacer()

                        Text(hiddenSummary)
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
    }

    private var hiddenSummary: String {
        engine.hiddenClasses.isEmpty
            ? "80 个类别全部显示"
            : "已隐藏 \(engine.hiddenClasses.count) 个类别"
    }

    private func binding(for index: Int) -> Binding<Bool> {
        Binding(
            get: { !engine.hiddenClasses.contains(index) },
            set: { isOn in
                if isOn {
                    engine.hiddenClasses.remove(index)
                } else {
                    engine.hiddenClasses.insert(index)
                }
            }
        )
    }
}

private struct ClassRow: Identifiable {
    let index: Int
    let name: String
    let english: String
    let isActive: Bool

    var id: Int { index }
}

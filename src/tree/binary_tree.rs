use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Serialize)]
#[repr(u8)]
pub enum SplitType {
    Numerical = 0,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[repr(C)]
pub(crate) struct DTNode {
    pub(crate) weight: f64,
    pub(crate) split_value: f64,
    pub(crate) feature_index: i32,
    pub(crate) is_leaf: bool,
    pub(crate) split_type: SplitType,
    pub(crate) default_left: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct BinaryTreeNode {
    pub(crate) value: DTNode,
    pub(crate) index: usize,
    pub(crate) left: usize,
    pub(crate) right: usize,
}

impl BinaryTreeNode {
    pub fn new(value: DTNode) -> Self {
        BinaryTreeNode {
            value,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

impl From<DTNode> for BinaryTreeNode {
    fn from(node: DTNode) -> Self {
        BinaryTreeNode {
            value: node,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BinaryTree {
    pub(crate) nodes: Vec<BinaryTreeNode>,
}

impl BinaryTree {
    pub(crate) fn new() -> Self {
        BinaryTree { nodes: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn get_root_index(&self) -> usize {
        0
    }

    pub(crate) fn get_node(&self, index: usize) -> Option<&BinaryTreeNode> {
        self.nodes.get(index)
    }

    #[allow(dead_code)]
    pub(crate) fn get_node_mut(&mut self, index: usize) -> Option<&mut BinaryTreeNode> {
        self.nodes.get_mut(index)
    }

    pub(crate) fn get_left_child(&self, node: &BinaryTreeNode) -> Option<&BinaryTreeNode> {
        if node.left == 0 {
            None
        } else {
            self.nodes.get(node.left)
        }
    }

    pub(crate) fn get_right_child(&self, node: &BinaryTreeNode) -> Option<&BinaryTreeNode> {
        if node.right == 0 {
            None
        } else {
            self.nodes.get(node.right)
        }
    }

    pub(crate) fn add_root(&mut self, mut root: BinaryTreeNode) -> usize {
        let index = self.nodes.len();
        root.index = index;
        self.nodes.push(root);
        index
    }

    pub(crate) fn add_left_node(&mut self, parent: usize, mut child: BinaryTreeNode) -> usize {
        let index = self.nodes.len();
        child.index = index;
        self.nodes.push(child);

        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.left = index;
        }
        index
    }

    pub(crate) fn add_right_node(&mut self, parent: usize, mut child: BinaryTreeNode) -> usize {
        let index = self.nodes.len();
        child.index = index;
        self.nodes.push(child);

        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.right = index;
        }
        index
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn add_orphan_node(&mut self, node: BinaryTreeNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    pub(crate) fn connect_left(&mut self, parent_idx: usize, child_idx: usize) -> Result<(), ()> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err(());
        }
        self.nodes[parent_idx].left = child_idx;
        Ok(())
    }

    pub(crate) fn connect_right(&mut self, parent_idx: usize, child_idx: usize) -> Result<(), ()> {
        if parent_idx >= self.nodes.len() || child_idx >= self.nodes.len() {
            return Err(());
        }
        self.nodes[parent_idx].right = child_idx;
        Ok(())
    }

    pub(crate) fn validate_connections(&self) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![0]; // Start from root

        while let Some(idx) = stack.pop() {
            visited[idx] = true;

            let node = &self.nodes[idx];
            if !node.value.is_leaf {
                if node.left >= self.nodes.len() || node.right >= self.nodes.len() {
                    return false;
                }
                stack.push(node.left);
                stack.push(node.right);
            }
        }

        visited.into_iter().all(|v| v)
    }
}

impl Default for BinaryTree {
    fn default() -> Self {
        Self::new()
    }
}

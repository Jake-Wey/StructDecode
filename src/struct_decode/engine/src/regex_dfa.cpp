#include "regex_dfa.h"
#include <stdexcept>

namespace struct_decode {

// ===========================================================================
// DFA Base Implementation
// ===========================================================================

DFA::DFA() : current_state_(START_STATE) {
	transitions_.emplace_back();
}

void DFA::reset() {
	current_state_ = START_STATE;
}

DFAState DFA::advance(char c) {
	if (current_state_ == DEAD_STATE) {
		return DEAD_STATE;
	}

	const auto& trans = transitions_[current_state_];
	auto it = trans.find(c);

	if (it == trans.end()) {
		current_state_ = DEAD_STATE;
	} else {
		current_state_ = it->second;
	}

	return current_state_;
}

bool DFA::is_accepting() const {
	return accepting_states_.count(current_state_) > 0;
}

bool DFA::is_dead() const {
	return current_state_ == DEAD_STATE;
}

std::unordered_set<char> DFA::get_valid_next_chars() const {
	return get_valid_next_chars(current_state_);
}

std::unordered_set<char> DFA::get_valid_next_chars(DFAState state) const {
	std::unordered_set<char> chars;

	if (state == DEAD_STATE || state < 0 || static_cast<size_t>(state) >= transitions_.size()) {
		return chars;
	}

	for (const auto& [c, target] : transitions_[state]) {
		if (target != DEAD_STATE) {
			chars.insert(c);
		}
	}

	return chars;
}

void DFA::add_state() {
	transitions_.emplace_back();
}

void DFA::add_transition(DFAState from, char c, DFAState to) {
	if (from >= 0 && static_cast<size_t>(from) < transitions_.size()) {
		transitions_[from][c] = to;
	}
}

void DFA::set_accepting(DFAState state) {
	accepting_states_.insert(state);
}

// ============================================================================
// LiteralDFA Implementation
// ============================================================================

LiteralDFA::LiteralDFA(const std::string& literal) {
	if (literal.empty()) {
		set_accepting(START_STATE);
		return;
	}

	DFAState prev = START_STATE;

	for (size_t i = 0; i < literal.size(); ++i) {
		add_state();
		DFAState next = static_cast<DFAState>(i + 1);
		add_transition(prev, literal[i], next);
		prev = next;
	}

	set_accepting(prev);
}

// ============================================================================
// CharClassDFA Implementation
// ============================================================================

CharClassDFA::CharClassDFA(const std::string& chars, bool negated) {
	add_state();
	DFAState accept_state = 1;
	DFAState reject_state = DEAD_STATE;

	if (negated) {
		for (int c = 32; c < 127; ++c) {
			if (chars.find(static_cast<char>(c)) == std::string::npos) {
				add_transition(START_STATE, static_cast<char>(c), accept_state);
			}
		}
	} else {
		for (char c : chars) {
			add_transition(START_STATE, c, accept_state);
		}
	}

	set_accepting(accept_state);
}

// ============================================================================
// RepeatingCharClassDFA Implementation
// ============================================================================

RepeatingCharClassDFA::RepeatingCharClassDFA(
	const std::string& chars, 
	int min_reps, 
	int max_reps
) {
	if (max_reps == -1) {
		// Unlimited repetitions
		add_state();
		DFAState accept_state = 1;

		for (char c : chars) {
			add_transition(START_STATE, c, accept_state);
			add_transition(accept_state, c, accept_state);
		}

		if (min_reps == 0) {
			set_accepting(START_STATE);
		}
		set_accepting(accept_state);
	} else {
		DFAState prev = START_STATE;

		for (int i = 0; i < max_reps; ++i) {
			add_state();
			DFAState next = static_cast<DFAState>(i + 1);

			for (char c : chars) {
				add_transition(prev, c, next);
			}

			if (i >= min_reps) {
				for (char c : chars) {
					add_transition(prev, c, next);
				}
			}

			if (i + 1 >= min_reps) {
				set_accepting(next);
			}

			prev = next;
		}

		if (min_reps == 0) {
			set_accepting(START_STATE);
		}
	}
}

// ============================================================================
// RegexDFA Implementation (Simplified Parser)
// ============================================================================

std::unique_ptr<DFA> RegexDFA::parse(const std::string& pattern) {
	if (pattern.empty()) {
		return std::make_unique<DFA>();
	}

	bool is_literal = true;
	for (char c : pattern) {
		if (c == '[' || c == ']' || c == '*' || c == '+' || c == '?' ||
			c == '|' || c == '(' || c == ')' || c == '.' || c == '\\') {
			is_literal = false;
			break;
		}
	}

	if (is_literal) {
		return std::make_unique<LiteralDFA>(pattern);
	}

	if (pattern.size() >= 3 && pattern[0] == '[' && (pattern.back() == '+' || pattern.back() == '*')) {
		size_t bracket_end = pattern.find(']');
		if (bracket_end != std::string::npos && bracket_end + 1 == pattern.size() - 1) {
			std::string char_class = pattern.substr(1, bracket_end - 1);

			std::string expanded;
			for (size_t i = 0; i < char_class.size(); ++i) {
				if (i + 2 < char_class.size() && char_class[i + 1] == '-') {
					char start = char_class[i];
					char end = char_class[i + 2];
					for (char c = start; c <= end; ++c) {
						expanded += c;
					}
					i += 2;
				} else {
					expanded += char_class[i];
				}
			}

			if (pattern.back() == '+') {
				return std::make_unique<RepeatingCharClassDFA>(expanded, 1, -1);
			} else {
				return std::make_unique<RepeatingCharClassDFA>(expanded, 0, -1);
			}
		}
	}

	throw std::runtime_error("Unsupported regex pattern: " + pattern);
}

bool RegexDFA::is_supported(const std::string& pattern) {
	try {
		parse(pattern);
		return true;
	} catch (...) {
		return false;
	}
}

} // struct_decode
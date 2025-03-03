import mne
from attrs import define, field

class ReportProxy:
    def __init__(self, report, owner_type):
        self._report = report
        self._owner_type = owner_type

    def __getattr__(self, name):
        attr = getattr(self._report, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                # Replace any argument that is an instance of the owner with its underlying raw.
                new_args = [
                    arg.raw if isinstance(arg, self._owner_type) else arg 
                    for arg in args
                ]
                new_kwargs = {
                    k: (v.raw if isinstance(v, self._owner_type) else v)
                    for k, v in kwargs.items()
                }
                return attr(*new_args, **new_kwargs)
            return wrapper
        return attr

# A mixin that creates and exposes the report via the proxy.
@define(kw_only=True)
class ReportMixin:
    _report: mne.Report = field(factory=mne.Report, init=False)

    @property
    def report(self):
        # Return a proxy that knows how to unwrap our wrapper type.
        return ReportProxy(self._report, type(self))

@define(kw_only=True)
class AlmkanalRaw(ReportMixin):
    raw: mne.io.Raw

    @classmethod
    def from_mne_raw(cls, raw: mne.io.Raw):
        cur_cls = cls(raw=raw)
        if cur_cls.report.__len__() == 0:
            cur_cls.report.add_raw(cur_cls, 'raw_data',
                                   butterfly=False, psd=True)
        return cur_cls

    def __getattr__(self, name):
        return getattr(self.raw, name)


@define(kw_only=True)
class AlmkanalEpochsMixin(ReportMixin):
    def __init__(self, raw: ReportMixin, *args, **kwargs):
        self.report = raw.report.copy()
        super().__init__(*args, **kwargs)


@define(kw_only=True)
class AlmkanalEpochs(mne.Epochs, AlmkanalEpochsMixin):
    pass